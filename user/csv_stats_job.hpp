#pragma once

#include "job_interface.hpp"

#include <cstdint>
#include <string>
#include <vector>

// One record emitted per CSV data row (not per file).
// Trivially copyable — required by SpscRing static_assert.
struct CsvRecord {
    long bytes;
    long words;
    long lines;
};

// stat_id constants for RankedEntry
constexpr uint8_t STAT_BYTES = 0;
constexpr uint8_t STAT_WORDS = 1;
constexpr uint8_t STAT_LINES = 2;

// direction constants for RankedEntry
constexpr uint8_t DIR_BOTTOM = 0;
constexpr uint8_t DIR_TOP    = 1;

constexpr std::size_t ranked_website_max_len = 64;

// One record emitted per heap survivor (up to 10 per stat per direction per worker).
// Carries the website name, the stat value, and which stat/direction this entry represents.
// Trivially copyable — required by SpscRing static_assert.
struct RankedEntry {
    char    website[ranked_website_max_len];  // 64 bytes — null-terminated
    uint8_t stat_id;                          //  1 byte
    uint8_t direction;                        //  1 byte
    char    padding[6];                       //  6 bytes explicit (avoids implicit padding before value)
    long    value;                            //  8 bytes
    // total: 80 bytes
};
static_assert(std::is_trivially_copyable_v<RankedEntry>);

// Result-only type for storing ranked websites on the job object.
// Not a ring type — non-trivial copyability is fine here.
struct RankedWebsite {
    std::string website;
    long        value;
};

class CsvStatsJob : public JobInterface<CsvRecord, RankedEntry> {
public:
    // Planning state (populated by on_cmdline / on_dist_plan / on_node_plan)
    std::string data_dir_;
    int         first_worker_id_{0};               // global ID of thread 0 on this rank
    std::vector<std::string>              rank_files_;    // this rank's CSV files
    std::vector<std::vector<std::string>> thread_files_;  // per-thread subdivision

    // Aggregate results written by reduce tasks, read by on_output
    long   record_count_ = 0;
    long   total_bytes_  = 0;
    long   total_words_  = 0;
    long   total_lines_  = 0;
    double avg_bytes_    = 0;
    double avg_words_    = 0;
    double avg_lines_    = 0;
    long   med_bytes_    = 0;
    long   med_words_    = 0;
    long   med_lines_    = 0;

    // Ranking results written by reduce tasks, read by on_output
    std::vector<RankedWebsite> top10_bytes_,   bottom10_bytes_;
    std::vector<RankedWebsite> top10_words_,   bottom10_words_;
    std::vector<RankedWebsite> top10_lines_,   bottom10_lines_;

    void on_cmdline   (CmdlineCtx&)                         override;
    void on_dist_plan (DistPlanCtx&)                        override;
    void on_node_plan (NodePlanCtx&)                        override;
    void on_map       (WorkCtx<CsvRecord, RankedEntry>&)    override;
    void on_reduce    (ReduceCtx<CsvRecord, RankedEntry>&)  override;
    void on_output    (OutputCtx&)                          override;
};
