#!/usr/bin/env python3
"""Generate deterministic CSV test data and expected output for the E2E test.

Produces tests/e2e/data/ (20 CSV files) and tests/e2e/expected.txt.
Safe to re-run: overwrites existing files.

Design constraints satisfied:
  - All bytes/words/lines values are distinct (no ties in median or rankings).
  - At least one file has a single data row.
  - Total row count is 41 (odd), so the median is the exact middle element.
  - Values are fixed integers — no randomness or seed dependency.
"""

import os
import pathlib

# ── Data layout ───────────────────────────────────────────────────────────────

# 20 CSV files; row count per file sums to 41 (odd).
# File 0 has exactly 1 row to exercise the single-row edge case.
ROWS_PER_FILE = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4]
assert len(ROWS_PER_FILE) == 20
assert sum(ROWS_PER_FILE) == 41

# Row i: values are unique across all rows (no ties), spaced so medians are exact integers.
#   bytes[i] = 100*i + 1   → values 1, 101, 201, ..., 4001
#   words[i] = 50*i  + 1   → values 1,  51, 101, ..., 2001
#   lines[i] = 10*i  + 1   → values 1,  11,  21, ...,  401
TOTAL_ROWS = sum(ROWS_PER_FILE)

def row_website(i: int) -> str: return f"site{i:03d}.com"
def row_bytes(i: int) -> int:   return 100 * i + 1
def row_words(i: int) -> int:   return  50 * i + 1
def row_lines(i: int) -> int:   return  10 * i + 1

# ── Generate CSV files ────────────────────────────────────────────────────────

script_dir = pathlib.Path(__file__).parent
data_dir   = script_dir / "data"
data_dir.mkdir(exist_ok=True)

row_index = 0
for file_num, n_rows in enumerate(ROWS_PER_FILE):
    path = data_dir / f"file{file_num:02d}.csv"
    with open(path, "w") as f:
        f.write("website,bytecount,wordcount,linecount\n")
        for _ in range(n_rows):
            b = row_bytes(row_index)
            w = row_words(row_index)
            l = row_lines(row_index)
            site = row_website(row_index)
            f.write(f"{site},{b},{w},{l}\n")
            row_index += 1

assert row_index == TOTAL_ROWS

# ── Compute expected aggregate results ────────────────────────────────────────

all_websites = [row_website(i) for i in range(TOTAL_ROWS)]
all_bytes    = [row_bytes(i)   for i in range(TOTAL_ROWS)]
all_words    = [row_words(i)   for i in range(TOTAL_ROWS)]
all_lines    = [row_lines(i)   for i in range(TOTAL_ROWS)]

total_bytes = sum(all_bytes)
total_words = sum(all_words)
total_lines = sum(all_lines)

avg_bytes = total_bytes / TOTAL_ROWS
avg_words = total_words / TOTAL_ROWS
avg_lines = total_lines / TOTAL_ROWS

# exact_median convention: lower of the two middle elements for even-length sequences.
# Index formula: (n-1)//2 — same convention as the C++ exact_median helper.
def exact_median(values: list) -> int:
    s = sorted(values)
    return s[(len(s) - 1) // 2]

med_bytes = exact_median(all_bytes)
med_words = exact_median(all_words)
med_lines = exact_median(all_lines)

# ── Compute expected rankings ─────────────────────────────────────────────────

# Sort ascending by (value, website) — matches C++ comparator.
# Top 10 = last 10 reversed (highest first).
# Bottom 10 = first 10 (lowest first).
# This mirrors the C++ build_rankings logic in on_reduce.
def top_bottom_10(values: list, websites: list):
    pairs = sorted(zip(values, websites))   # ascending (value, website)
    take  = min(10, len(pairs))
    bottom = pairs[:take]
    top    = list(reversed(pairs[-take:]))
    return top, bottom  # each element is (value, website)

top_bytes,   bot_bytes   = top_bottom_10(all_bytes, all_websites)
top_words,   bot_words   = top_bottom_10(all_words, all_websites)
top_lines,   bot_lines   = top_bottom_10(all_lines, all_websites)

# ── Write expected.txt ────────────────────────────────────────────────────────

expected_path = script_dir / "expected.txt"
with open(expected_path, "w") as f:
    f.write(f"records={TOTAL_ROWS}\n")

    def write_stat(label, total, avg, median, top, bottom):
        f.write(f"{label}  total={total} avg={avg:.3f} median={median}\n")
        f.write(f"{'%-5s' % label}  top10:\n")
        for val, site in top:
            f.write(f"  {site}={val}\n")
        f.write(f"{'%-5s' % label}  bottom10:\n")
        for val, site in bottom:
            f.write(f"  {site}={val}\n")

    write_stat("bytes", total_bytes, avg_bytes, med_bytes, top_bytes, bot_bytes)
    write_stat("words", total_words, avg_words, med_words, top_words, bot_words)
    write_stat("lines", total_lines, avg_lines, med_lines, top_lines, bot_lines)

print(f"Generated {len(ROWS_PER_FILE)} CSV files in {data_dir}/")
print(f"Written {expected_path}")
print(f"  records={TOTAL_ROWS}")
print(f"  bytes  total={total_bytes} avg={avg_bytes:.3f} median={med_bytes}")
print(f"  words  total={total_words} avg={avg_words:.3f} median={med_words}")
print(f"  lines  total={total_lines} avg={avg_lines:.3f} median={med_lines}")
