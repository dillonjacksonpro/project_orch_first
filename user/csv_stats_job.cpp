#include "csv_stats_job.hpp"

#include "context/cmdline_ctx.hpp"
#include "context/dist_plan_ctx.hpp"
#include "context/node_plan_ctx.hpp"
#include "context/output_ctx.hpp"
#include "context/reduce_ctx.hpp"
#include "context/work_ctx.hpp"
#include "fatal.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ── Helpers ───────────────────────────────────────────────────────────────────

// Intermediate parse result — includes website, not trivially copyable, never enters the ring.
struct ParsedRow {
    std::string website;
    long bytes;
    long words;
    long lines;
};

// Parse a single CSV data row with format: website,bytes,words,lines
// Returns std::nullopt silently for malformed rows (wrong field count, non-integer values).
[[nodiscard]] static std::optional<ParsedRow> parse_row(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    if (fields.size() != 4) return std::nullopt;
    try {
        long bytes = std::stol(fields[1]);
        long words = std::stol(fields[2]);
        long lines = std::stol(fields[3]);
        return ParsedRow{fields[0], bytes, words, lines};
    } catch (...) {
        return std::nullopt;
    }
}

// Compute the exact median of v using nth_element.
// Convention: lower of the two middle elements for even-length sequences.
// Index formula: (n-1)/2 — works for both odd and even n:
//   n=1 → 0, n=2 → 0 (lower), n=3 → 1, n=4 → 1 (lower), n=5 → 2
[[nodiscard]] static long exact_median(std::vector<long>& v) {
    const std::size_t mid = (v.size() - 1) / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
    return v[mid];
}

// ── Hook implementations ───────────────────────────────────────────────────────

void CsvStatsJob::on_cmdline(CmdlineCtx& ctx) {
    for (int i = 1; i < ctx.argc - 1; ++i) {
        if (std::string(ctx.argv[i]) == "--dir") {
            data_dir_ = ctx.argv[i + 1];
            return;
        }
    }
    // FATAL is single-process here (pre-MPI); use the non-MPI variant.
    FATAL("--dir <path> argument is required");
}

void CsvStatsJob::on_dist_plan(DistPlanCtx& ctx) {
    // Capture this rank's first global worker ID for use in on_map.
    first_worker_id_ = ctx.first_global_worker_id;

    // Enumerate all *.csv files and sort for determinism across ranks.
    std::vector<std::string> all_files;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir_)) {
        if (entry.path().extension() == ".csv") {
            all_files.push_back(entry.path().string());
        }
    }
    std::sort(all_files.begin(), all_files.end());

    // Stride-partition: file i goes to rank (i % world_size).
    for (int i = 0; i < static_cast<int>(all_files.size()); ++i) {
        if (i % ctx.world_size == ctx.rank) {
            rank_files_.push_back(all_files[static_cast<std::size_t>(i)]);
        }
    }
}

void CsvStatsJob::on_node_plan(NodePlanCtx& ctx) {
    // Stride-partition rank_files_ across threads: file i goes to thread (i % num_workers).
    thread_files_.resize(static_cast<std::size_t>(ctx.num_workers));
    for (int i = 0; i < static_cast<int>(rank_files_.size()); ++i) {
        const std::size_t tid = static_cast<std::size_t>(i % ctx.num_workers);
        thread_files_[tid].push_back(rank_files_[static_cast<std::size_t>(i)]);
    }
}

void CsvStatsJob::on_map(WorkCtx<CsvRecord, RankedEntry>& ctx) {
    // ctx.id is the global worker ID; subtract first_worker_id_ to get the local index
    // into thread_files_ (which is sized by num_workers on this rank).
    const int local_tid = ctx.id - first_worker_id_;

    // Top-K: min-heap of size K — evicts the smallest to keep the K largest.
    // Bottom-K: max-heap of size K — evicts the largest to keep the K smallest.
    // Key: pair<long, string> — comparisons are lexicographic (value first, website second),
    // which gives deterministic tie-breaking by website name when values are equal.
    using Entry = std::pair<long, std::string>;
    constexpr int k = 10;
    // min-heaps for top-K (std::greater makes the smallest element the top)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>>
        top_bytes, top_words, top_lines;
    // max-heaps for bottom-K (default std::less makes the largest element the top)
    std::priority_queue<Entry> bot_bytes, bot_words, bot_lines;

    // Push entry into a top-K min-heap; evict if over capacity.
    auto push_top = [k](auto& heap, long val, const std::string& web) {
        heap.emplace(val, web);
        if (static_cast<int>(heap.size()) > k) heap.pop();
    };
    // Push entry into a bottom-K max-heap; evict if over capacity.
    auto push_bot = [k](auto& heap, long val, const std::string& web) {
        heap.emplace(val, web);
        if (static_cast<int>(heap.size()) > k) heap.pop();
    };

    for (const auto& path : thread_files_[static_cast<std::size_t>(local_tid)]) {
        std::ifstream f(path);
        std::string line;
        std::getline(f, line); // skip header
        while (std::getline(f, line)) {
            auto parsed = parse_row(line);
            if (!parsed) continue;
            // Emit the full numeric record for aggregate stats (totals, avg, median).
            ctx.emit(CsvRecord{parsed->bytes, parsed->words, parsed->lines});
            // Update local heaps — website only lives in the heap, never in CsvRecord.
            push_top(top_bytes, parsed->bytes, parsed->website);
            push_bot(bot_bytes, parsed->bytes, parsed->website);
            push_top(top_words, parsed->words, parsed->website);
            push_bot(bot_words, parsed->words, parsed->website);
            push_top(top_lines, parsed->lines, parsed->website);
            push_bot(bot_lines, parsed->lines, parsed->website);
        }
    }

    // Drain all heap survivors into RankedEntry records and emit to root.
    // Heap emission order is arbitrary — root re-sorts when merging.
    auto emit_heap = [&](auto& heap, uint8_t stat, uint8_t dir) {
        while (!heap.empty()) {
            RankedEntry e{};
            std::snprintf(e.website, ranked_website_max_len, "%s",
                          heap.top().second.c_str());
            e.stat_id   = stat;
            e.direction = dir;
            e.value     = heap.top().first;
            ctx.emit(e);
            heap.pop();
        }
    };
    emit_heap(top_bytes, STAT_BYTES, DIR_TOP);
    emit_heap(bot_bytes, STAT_BYTES, DIR_BOTTOM);
    emit_heap(top_words, STAT_WORDS, DIR_TOP);
    emit_heap(bot_words, STAT_WORDS, DIR_BOTTOM);
    emit_heap(top_lines, STAT_LINES, DIR_TOP);
    emit_heap(bot_lines, STAT_LINES, DIR_BOTTOM);
}

void CsvStatsJob::on_reduce(ReduceCtx<CsvRecord, RankedEntry>& ctx) {
    // data: all CsvRecord entries from all workers (for aggregates).
    // ranked: all RankedEntry survivors from all workers' heaps (for rankings).
    auto data   = ctx.data<CsvRecord>();
    auto ranked = ctx.data<RankedEntry>();

    // ── Existing aggregate tasks ───────────────────────────────────────────────

    ctx.add_task("count_and_totals", [this, data]() {
        record_count_ = static_cast<long>(data.size);
        for (const auto& r : data) {
            total_bytes_ += r.bytes;
            total_words_ += r.words;
            total_lines_ += r.lines;
        }
        if (record_count_ > 0) {
            avg_bytes_ = static_cast<double>(total_bytes_) / record_count_;
            avg_words_ = static_cast<double>(total_words_) / record_count_;
            avg_lines_ = static_cast<double>(total_lines_) / record_count_;
        }
    });

    ctx.add_task("median_bytes", [this, data]() {
        std::vector<long> v(data.size);
        for (std::size_t i = 0; i < data.size; ++i) v[i] = data.data[i].bytes;
        med_bytes_ = exact_median(v);
    });

    ctx.add_task("median_words", [this, data]() {
        std::vector<long> v(data.size);
        for (std::size_t i = 0; i < data.size; ++i) v[i] = data.data[i].words;
        med_words_ = exact_median(v);
    });

    ctx.add_task("median_lines", [this, data]() {
        std::vector<long> v(data.size);
        for (std::size_t i = 0; i < data.size; ++i) v[i] = data.data[i].lines;
        med_lines_ = exact_median(v);
    });

    // ── Ranking tasks — one per stat, each writes to separate result fields ────

    // Helper: given per-worker candidates, sort and select global top/bottom 10.
    // Sorting key: (value ASC, website ASC) — one pass covers both extremes.
    // Top 10 = last 10 entries reversed (highest first).
    // Bottom 10 = first 10 entries (lowest first).
    auto build_rankings = [](
            const std::vector<std::pair<long, std::string>>& top_cands,
            const std::vector<std::pair<long, std::string>>& bot_cands,
            std::vector<RankedWebsite>& top_out,
            std::vector<RankedWebsite>& bot_out) {
        auto asc = [](const auto& a, const auto& b) {
            // sort ascending by (value, website) — deterministic tie-breaking
            return a.first != b.first ? a.first < b.first : a.second < b.second;
        };
        auto top_sorted = top_cands;
        auto bot_sorted = bot_cands;
        std::sort(top_sorted.begin(), top_sorted.end(), asc);
        std::sort(bot_sorted.begin(), bot_sorted.end(), asc);

        const int take_t = std::min(10, static_cast<int>(top_sorted.size()));
        const int take_b = std::min(10, static_cast<int>(bot_sorted.size()));
        top_out.reserve(take_t);
        bot_out.reserve(take_b);
        // Top: iterate from the back (highest first)
        for (int i = static_cast<int>(top_sorted.size()) - 1;
             i >= static_cast<int>(top_sorted.size()) - take_t; --i) {
            top_out.push_back({top_sorted[static_cast<std::size_t>(i)].second,
                               top_sorted[static_cast<std::size_t>(i)].first});
        }
        // Bottom: iterate from the front (lowest first)
        for (int i = 0; i < take_b; ++i) {
            bot_out.push_back({bot_sorted[static_cast<std::size_t>(i)].second,
                               bot_sorted[static_cast<std::size_t>(i)].first});
        }
    };

    ctx.add_task("rankings_bytes", [this, ranked, build_rankings]() {
        std::vector<std::pair<long, std::string>> top_cands, bot_cands;
        for (const auto& e : ranked) {
            if (e.stat_id != STAT_BYTES) continue;
            if (e.direction == DIR_TOP) top_cands.emplace_back(e.value, std::string(e.website));
            else                        bot_cands.emplace_back(e.value, std::string(e.website));
        }
        build_rankings(top_cands, bot_cands, top10_bytes_, bottom10_bytes_);
    });

    ctx.add_task("rankings_words", [this, ranked, build_rankings]() {
        std::vector<std::pair<long, std::string>> top_cands, bot_cands;
        for (const auto& e : ranked) {
            if (e.stat_id != STAT_WORDS) continue;
            if (e.direction == DIR_TOP) top_cands.emplace_back(e.value, std::string(e.website));
            else                        bot_cands.emplace_back(e.value, std::string(e.website));
        }
        build_rankings(top_cands, bot_cands, top10_words_, bottom10_words_);
    });

    ctx.add_task("rankings_lines", [this, ranked, build_rankings]() {
        std::vector<std::pair<long, std::string>> top_cands, bot_cands;
        for (const auto& e : ranked) {
            if (e.stat_id != STAT_LINES) continue;
            if (e.direction == DIR_TOP) top_cands.emplace_back(e.value, std::string(e.website));
            else                        bot_cands.emplace_back(e.value, std::string(e.website));
        }
        build_rankings(top_cands, bot_cands, top10_lines_, bottom10_lines_);
    });
}

void CsvStatsJob::on_output(OutputCtx& /*ctx*/) {
    // Write to stdout — the E2E script redirects stdout for diffing and discards stderr.
    auto print_list = [](const char* stat, const char* tag,
                         const std::vector<RankedWebsite>& v) {
        std::printf("%-5s  %s:\n", stat, tag);
        for (const auto& e : v) {
            std::printf("  %s=%ld\n", e.website.c_str(), e.value);
        }
    };

    std::printf("records=%ld\n", record_count_);

    std::printf("bytes  total=%ld avg=%.3f median=%ld\n", total_bytes_, avg_bytes_, med_bytes_);
    print_list("bytes", "top10",    top10_bytes_);
    print_list("bytes", "bottom10", bottom10_bytes_);

    std::printf("words  total=%ld avg=%.3f median=%ld\n", total_words_, avg_words_, med_words_);
    print_list("words", "top10",    top10_words_);
    print_list("words", "bottom10", bottom10_words_);

    std::printf("lines  total=%ld avg=%.3f median=%ld\n", total_lines_, avg_lines_, med_lines_);
    print_list("lines", "top10",    top10_lines_);
    print_list("lines", "bottom10", bottom10_lines_);
}
