#!/usr/bin/env bash
# mini_eval_v2.sh
# ---------------
# Run the coverage-driven expanded evaluation benchmark (mini_eval_v2).
#
# This is the BROADER validation benchmark — 84 scenarios across 8 coverage
# buckets. It is NOT a fast-iteration benchmark. Use script/mini_benchmark.sh
# for rapid dev cycles.
#
# Usage:
#   bash script/mini_eval_v2.sh [challenge]
#
# challenge defaults to closed_loop_nonreactive_agents.
# Other options:
#   closed_loop_reactive_agents
#   open_loop_boxes
#
# Outputs:
#   datasets/nuplan/exp/exp/simulation/<challenge>/<timestamp>/
#   Aggregated metrics and per-scenario scores auto-printed at end.
#
# After the run, summarize results:
#   python script/summarize_eval.py --challenge <challenge>
#
# See docs/EVAL_PLAN.md for benchmark philosophy.
# See docs/mini_scenario_inventory.md for scenario selection rationale.

set -euo pipefail

CHALLENGE="${1:-closed_loop_nonreactive_agents}"
cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
PLANNER="planTF"

# Validate challenge
case "$CHALLENGE" in
    closed_loop_nonreactive_agents|closed_loop_reactive_agents|open_loop_boxes)
        ;;
    *)
        echo "ERROR: Unknown challenge '$CHALLENGE'"
        echo "       Valid: closed_loop_nonreactive_agents | closed_loop_reactive_agents | open_loop_boxes"
        exit 1
        ;;
esac

echo "========================================================"
echo "  planTF mini_eval_v2 — Coverage-Driven Benchmark"
echo "========================================================"
echo "  Challenge:  $CHALLENGE"
echo "  Checkpoint: $CKPT_ROOT/$PLANNER.ckpt"
echo "  Scenarios:  config/scenario_filter/mini_eval_v2.yaml (84 scenarios)"
echo "  Buckets:    intersections, lane_change, lead_vehicle, pedestrian,"
echo "              stop_and_go, high_medium_speed, near_long_vehicle, edge_cases"
echo ""
echo "  NOTE: This benchmark takes significantly longer than mini_benchmark.sh."
echo "        Expect ~8–15 minutes depending on hardware."
echo "  NOTE: Lane-change coverage is limited by mini-split availability (7 tokens total)."
echo "========================================================"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export PYTHONPATH="$cwd:${PYTHONPATH:-}"

python run_simulation.py \
    +simulation="$CHALLENGE" \
    planner="$PLANNER" \
    scenario_builder=nuplan \
    scenario_filter=mini_eval_v2 \
    worker.threads_per_node=8 \
    experiment_uid="mini_eval_v2/$PLANNER/$TIMESTAMP" \
    verbose=true \
    planner.imitation_planner.planner_ckpt="$CKPT_ROOT/$PLANNER.ckpt"

echo ""
echo "========================================================"
echo "  Simulation complete."
echo ""
echo "  To summarize results by bucket:"
echo "    python script/summarize_eval.py \\"
echo "      --challenge $CHALLENGE \\"
echo "      --v1-dir  benchmarks/mini_v1"
echo "========================================================"
