#!/bin/bash

# Benchmark run_planner.py w trzech trybach:
#   linear, mlp, both
#
# Wyniki lecą do:
#   bench/<tryb>/<scenariusz>/iter_<N>/
#
# Skrypt raportuje metryki neutralnie: czas, błąd TCP, margines i success.
# Nie zakłada, że MLP jest zawsze lepszy; przewagę należy czytać z CSV.

set -u

BENCH_DIR="bench"
OUT="$BENCH_DIR/benchmark_results.txt"
METRICS_CSV="$BENCH_DIR/benchmark_metrics.csv"
SUMMARY_CSV="$BENCH_DIR/benchmark_summary.csv"
MLP_CHECKPOINT="warmstart_mlp_unroll.pt"

MODES=(linear mlp both)

SCENARIOS=(
    s2_central_obstacle
    s3_narrow_passage
    s4_goal_near_obstacle
    s5_around_back
    s7_low_reach
    s8_clutter
)

ITERS=(5 10 15 20 25 30 35 40 45 50 70 100)

mkdir -p "$BENCH_DIR"

{
    echo "=== Benchmark run_planner.py ==="
    echo "Data startu: $(date)"
    echo "Checkpoint MLP: $MLP_CHECKPOINT"
    echo "Folder wyników: $BENCH_DIR"
    echo "Tryby: ${MODES[*]}"
    echo "Scenariusze: ${SCENARIOS[*]}"
    echo "Iteracje: ${ITERS[*]}"
    echo "Metryka success: tcp_error_m < 0.005 oraz final_margin_m >= 0.0"
    echo "Uwaga: benchmark nie zakłada, że MLP zawsze jest lepszy; porównuj osobno czas, TCP, clearance i success_rate."
    echo ""
} > "$OUT"

clean_old_outputs() {
    rm -f *.png *.npz run_metrics.csv comparison_metrics.csv
    rm -f data/*.csv
    rm -f data/figures/*
}

copy_outputs() {
    local run_dir="$1"

    for file in *.png *.npz run_metrics.csv comparison_metrics.csv; do
        if [ -f "$file" ]; then
            cp "$file" "$run_dir/"
        fi
    done

    if [ -d "data" ]; then
        find data -maxdepth 1 -type f -name "*.csv" -exec cp {} "$run_dir/" \;
    fi

    if [ -d "data/figures" ]; then
        find data/figures -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.pdf" \) \
            -exec cp {} "$run_dir/" \;
    fi
}

run_one() {
    local mode="$1"
    local scene="$2"
    local iter="$3"
    local run_dir="$BENCH_DIR/$mode/$scene/iter_$iter"
    local run_log="$run_dir/run.log"

    mkdir -p "$run_dir"

    echo "Uruchamiam: mode=$mode, scene=$scene, iter=$iter"

    {
        echo ""
        echo "=================================================="
        echo "MODE: $mode"
        echo "SCENARIO: $scene"
        echo "ITERATIONS: $iter"
        echo "RUN_DIR: $run_dir"
        echo "=================================================="
    } >> "$OUT"

    clean_old_outputs

    if [ "$mode" = "linear" ]; then
        python3 run_planner.py \
            --warmstart-mode linear \
            --scenario "$scene" \
            --max-iterations "$iter" \
            > "$run_log" 2>&1
    else
        python3 run_planner.py \
            --warmstart-mode "$mode" \
            --scenario "$scene" \
            --mlp-checkpoint "$MLP_CHECKPOINT" \
            --max-iterations "$iter" \
            > "$run_log" 2>&1
    fi

    status=$?
    cat "$run_log" >> "$OUT"
    echo "EXIT_CODE: $status" >> "$OUT"

    copy_outputs "$run_dir"

    {
        echo "MODE=$mode"
        echo "SCENARIO=$scene"
        echo "ITERATIONS=$iter"
        echo "CHECKPOINT=$MLP_CHECKPOINT"
        echo "EXIT_CODE=$status"
        echo "DATE=$(date)"
    } > "$run_dir/run_info.txt"

    if [ "$status" -ne 0 ]; then
        echo "UWAGA: błąd dla mode=$mode, scene=$scene, iter=$iter. Sprawdź $run_log"
    fi
}

aggregate_metrics() {
    python3 - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

bench = Path("bench")
metrics_out = bench / "benchmark_metrics.csv"
summary_out = bench / "benchmark_summary.csv"

rows = []
for metrics_path in sorted(bench.glob("*/*/iter_*/run_metrics.csv")):
    parts = metrics_path.parts
    run_mode = parts[-4]
    scenario = parts[-3]
    iter_name = parts[-2]
    try:
        iterations = int(iter_name.split("_", 1)[1])
    except Exception:
        iterations = ""
    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "run_mode": run_mode,
                "scenario": scenario,
                "iterations": iterations,
                **row,
            })

if not rows:
    print("Brak run_metrics.csv do agregacji.")
    raise SystemExit(0)

keys = []
seen = set()
for row in rows:
    for key in row.keys():
        if key not in seen:
            keys.append(key)
            seen.add(key)

with metrics_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)

summary_groups = defaultdict(list)
for row in rows:
    summary_groups[(row["run_mode"], row["method"])].append(row)

summary = []
for (run_mode, method), items in sorted(summary_groups.items()):
    n = len(items)
    def mean(key):
        vals = [float(r[key]) for r in items if r.get(key) not in (None, "")]
        return sum(vals) / len(vals) if vals else ""
    success_count = sum(int(float(r.get("success", 0))) for r in items)
    summary.append({
        "run_mode": run_mode,
        "method": method,
        "num_runs": n,
        "success_count": success_count,
        "success_rate": success_count / n if n else 0.0,
        "mean_time_s": mean("time_s"),
        "mean_tcp_error_m": mean("tcp_error_m"),
        "mean_final_margin_m": mean("final_margin_m"),
    })

with summary_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

print(f"Zapisano agregat: {metrics_out}")
print(f"Zapisano podsumowanie: {summary_out}")
PY
}

for mode in "${MODES[@]}"; do
    for scene in "${SCENARIOS[@]}"; do
        for iter in "${ITERS[@]}"; do
            run_one "$mode" "$scene" "$iter"
        done
    done
done

aggregate_metrics >> "$OUT" 2>&1

{
    echo ""
    echo "=================================================="
    echo "KONIEC BENCHMARKU"
    echo "Data końca: $(date)"
    echo "Metryki zbiorcze: $METRICS_CSV"
    echo "Podsumowanie: $SUMMARY_CSV"
    echo "=================================================="
} >> "$OUT"

echo ""
echo "Gotowe."
echo "Log zbiorczy: $OUT"
echo "Metryki zbiorcze: $METRICS_CSV"
echo "Podsumowanie: $SUMMARY_CSV"
echo "Foldery wyników: $BENCH_DIR/<linear|mlp|both>/<scenario>/iter_<N>/"
