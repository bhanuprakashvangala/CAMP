#!/usr/bin/env python3
"""Paper-faithful Experiment 1 gating logic for point-prediction models.

Implements the Section IV-A behavior from the paper:

1. Novelty score N(t):
   normalized distance from the nearest previously observed input regime in
   the same (workflow, process) bucket, using log1p(a_MB) as the static
   feature available at submission time.
2. Risk score R(t):
   empirical P(M > cap(t)) estimated from the bucket's historical residual
   distribution in log space, where residual = log(M) - log(predicted).
3. Composite score:
   S(t) = w_R * R(t) + w_N * N(t)
4. Audit decision:
   Audit when S(t) > tau_t, where tau_t is updated online via a PID-style
   controller to maintain the target audit rate B.

The script emits a detailed per-task scored CSV and a wrapper-ready task-plan
CSV compatible with the existing selective eBPF audit tooling.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass


DEFAULT_PREDICTED_MEMORY_COLUMN = "pred_lgbm_sizey_MB"
DEFAULT_SAFE_MEMORY_COLUMN = "safe_lgbm_sizey_MB"
HISTORICAL_MEMORY_COLUMN = "M_MB"
DEFAULT_PREDICTION_MODEL_NAME = "point_prediction_section_iv_a"

ENGINE_TASK_INSTANCE_COLUMNS = {
    "nextflow": ("task_hash", "task_id", "native_id", "tag", "name", "sample", "hash"),
    "snakemake": ("task_id", "tag", "sample", "name", "native_id", "hash", "task_hash"),
    "pegasus": ("task_id", "name", "native_id", "tag", "sample", "hash", "task_hash"),
    "generic": ("task_id", "task_hash", "native_id", "tag", "name", "sample", "hash"),
}

EPS = 1e-6


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


configure_csv_field_limit()


@dataclass
class ScoredTask:
    row_index: int
    workflow: str
    process: str
    engine: str
    task_type: str
    task_instance: str
    task_instance_base: str
    task_identity_source: str
    task_hash: str
    task_id: str
    a_mb: float
    c_mb: float
    m_mb: float | None
    predicted_memory_mb: float
    safe_memory_mb: float
    planned_memory_mb: int
    bucket_size_seen_before: int
    risk_history_size: int
    historical_memory_used: bool
    novelty_feature_log_a: float
    novelty_distance: float | None
    novelty_score: float
    risk_score: float
    log_cap_gap: float | None
    final_score: float
    selection_probability_pt: float
    importance_weight_1_over_pt: float
    tau_before: float
    tau_after: float
    pid_error: float
    running_audit_rate: float
    audit_flag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-faithful Experiment 1 gating (Section IV-A): novelty + risk "
            "composite with PID threshold control."
        )
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions_all.csv.")
    parser.add_argument("--out-plan", required=True, help="Output task-plan CSV path.")
    parser.add_argument("--out-scores", required=True, help="Output scored CSV path.")
    parser.add_argument(
        "--workflow",
        action="append",
        default=[],
        help="Optional exact workflow filter. Repeat to keep multiple workflows.",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "nextflow", "snakemake", "pegasus"),
        default="auto",
        help=(
            "How task instances are resolved. 'auto' prefers Nextflow-style hashes "
            "when present, otherwise falls back to Snakemake/Pegasus-friendly IDs."
        ),
    )
    parser.add_argument(
        "--predicted-column",
        default=DEFAULT_PREDICTED_MEMORY_COLUMN,
        help="Column containing the point-predicted peak memory in MB.",
    )
    parser.add_argument(
        "--safe-column",
        default=DEFAULT_SAFE_MEMORY_COLUMN,
        help="Column containing the safe/padded peak memory in MB.",
    )
    parser.add_argument(
        "--prediction-model-name",
        default=DEFAULT_PREDICTION_MODEL_NAME,
        help="Label written into output CSVs.",
    )
    parser.add_argument(
        "--novelty-scope",
        choices=("bucket", "global"),
        default="bucket",
        help="Compare novelty within the same (workflow, process) bucket or across all tasks.",
    )
    parser.add_argument(
        "--memory-for-plan",
        choices=("safe", "predicted", "max"),
        default="max",
        help=(
            "Memory written into the task plan. 'max' uses max(predicted, safe), "
            "which is the safest default for scheduler allocation."
        ),
    )
    parser.add_argument("--budget", type=float, default=0.10, help="Target global audit rate B.")
    parser.add_argument("--tau-init", type=float, default=0.50, help="Initial adaptive threshold tau_0.")
    parser.add_argument("--tau-min", type=float, default=0.0, help="Lower clamp for tau.")
    parser.add_argument("--tau-max", type=float, default=1.0, help="Upper clamp for tau.")
    parser.add_argument("--pid-kp", type=float, default=0.30, help="PID proportional gain.")
    parser.add_argument("--pid-ki", type=float, default=0.02, help="PID integral gain.")
    parser.add_argument("--pid-kd", type=float, default=0.05, help="PID derivative gain.")
    parser.add_argument("--pid-integral-min", type=float, default=-10.0, help="Lower clamp for PID integral state.")
    parser.add_argument("--pid-integral-max", type=float, default=10.0, help="Upper clamp for PID integral state.")
    parser.add_argument(
        "--cold-start-risk",
        choices=("neutral", "predicted_over_safe"),
        default="neutral",
        help=(
            "How to score risk when no historical residuals are available. "
            "'neutral' sets risk to 0.5. 'predicted_over_safe' uses predicted/safe."
        ),
    )
    parser.add_argument(
        "--novelty-scale",
        type=float,
        default=1.0,
        help="Normalization scale for nearest-neighbor distance in log1p(a_MB) space.",
    )
    parser.add_argument(
        "--novelty-history-limit",
        type=int,
        default=512,
        help="Maximum number of prior feature values kept per novelty history.",
    )
    parser.add_argument("--weight-risk", type=float, default=0.5, help="Weight of risk in the final score.")
    parser.add_argument("--weight-novelty", type=float, default=0.5, help="Weight of novelty in the final score.")
    parser.add_argument(
        "--selection-prob-temperature",
        type=float,
        default=0.05,
        help="Temperature for the smooth selection-probability approximation p_t.",
    )
    parser.add_argument(
        "--selection-prob-epsilon",
        type=float,
        default=1e-6,
        help="Lower clamp for p_t so 1/p_t stays finite.",
    )
    return parser.parse_args()


def normalize_text(value: str | None) -> str:
    return str(value or "").strip()


def parse_float(value: str | None, default: float = 0.0) -> float:
    text = normalize_text(value)
    if not text:
        return default
    return float(text)


def parse_optional_float(value: str | None) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    return float(text)


def build_task_type(row: dict[str, str]) -> str:
    for column in ("process", "task_type", "name", "workflow"):
        value = normalize_text(row.get(column))
        if value:
            return value
    return "unknown_task_type"


def infer_engine(row: dict[str, str]) -> str:
    if normalize_text(row.get("task_hash")):
        return "nextflow"
    if normalize_text(row.get("task_id")):
        return "snakemake"
    if normalize_text(row.get("native_id")) or normalize_text(row.get("name")):
        return "pegasus"
    return "generic"


def resolve_engine(requested_engine: str, row: dict[str, str]) -> str:
    if requested_engine == "auto":
        return infer_engine(row)
    return requested_engine


def resolve_task_instance_base(row: dict[str, str], engine: str, row_index: int) -> tuple[str, str]:
    columns = ENGINE_TASK_INSTANCE_COLUMNS.get(engine, ENGINE_TASK_INSTANCE_COLUMNS["generic"])
    for column in columns:
        value = normalize_text(row.get(column))
        if value:
            return value, column
    return f"row_{row_index:06d}", "row_index"


def ensure_unique_task_instance(
    used_plan_keys: dict[tuple[str, str], int],
    task_type: str,
    task_instance_base: str,
) -> str:
    key = (task_type, task_instance_base)
    duplicate_index = used_plan_keys.get(key, 0)
    used_plan_keys[key] = duplicate_index + 1
    if duplicate_index == 0:
        return task_instance_base
    return f"{task_instance_base}__dup{duplicate_index + 1}"


def append_bounded_history(history: list[float], value: float, history_limit: int) -> None:
    history.append(value)
    if history_limit > 0 and len(history) > history_limit:
        del history[0 : len(history) - history_limit]


def planned_memory_mb(predicted_memory_mb: float, safe_memory_mb: float, mode: str) -> int:
    if mode == "predicted":
        value = predicted_memory_mb
    elif mode == "safe":
        value = safe_memory_mb
    else:
        value = max(predicted_memory_mb, safe_memory_mb)
    return max(1, math.ceil(value))


def compute_novelty_score(
    feature_value: float,
    history: list[float],
    novelty_scale: float,
) -> tuple[float, float | None]:
    if not history:
        return 1.0, None
    min_distance = min(abs(feature_value - previous) for previous in history)
    score = min(min_distance / max(novelty_scale, EPS), 1.0)
    return score, min_distance


def compute_risk_score(
    residual_history: list[float],
    predicted_memory_mb: float,
    allocation_mb: float,
    safe_memory_mb: float,
    cold_start_policy: str,
) -> tuple[float, bool, float | None]:
    predicted = max(predicted_memory_mb, EPS)
    allocation = max(allocation_mb, EPS)
    if residual_history:
        log_cap_gap = math.log(allocation) - math.log(predicted)
        exceedances = sum(1 for residual in residual_history if residual > log_cap_gap)
        return exceedances / len(residual_history), True, log_cap_gap

    if cold_start_policy == "predicted_over_safe":
        safe = max(safe_memory_mb, predicted)
        return min(predicted / safe, 1.0), False, None

    return 0.5, False, None


def compute_selection_probability(
    final_score: float,
    tau_before: float,
    temperature: float,
    epsilon: float,
) -> float:
    temp = max(temperature, 1e-9)
    margin = (final_score - tau_before) / temp
    if margin >= 0:
        z = math.exp(-margin)
        p = 1.0 / (1.0 + z)
    else:
        z = math.exp(margin)
        p = z / (1.0 + z)
    return min(max(p, epsilon), 1.0)


def should_keep_row(
    row: dict[str, str],
    workflows: set[str],
    predicted_column: str,
    safe_column: str,
) -> bool:
    if workflows and normalize_text(row.get("workflow")) not in workflows:
        return False
    return bool(normalize_text(row.get(predicted_column)) and normalize_text(row.get(safe_column)))


def load_and_score_tasks(args: argparse.Namespace) -> tuple[list[ScoredTask], int]:
    workflows = set(args.workflow)
    bucket_feature_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    bucket_residual_history: dict[tuple[str, str], list[float]] = defaultdict(list)
    used_plan_keys: dict[tuple[str, str], int] = {}
    global_feature_history: list[float] = []
    skipped_rows = 0
    scored_tasks: list[ScoredTask] = []

    tau = min(max(args.tau_init, args.tau_min), args.tau_max)
    pid_integral = 0.0
    previous_error = 0.0
    audited_count = 0
    seen_count = 0

    with open(args.predictions, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            if not should_keep_row(
                row,
                workflows,
                predicted_column=args.predicted_column,
                safe_column=args.safe_column,
            ):
                skipped_rows += 1
                continue

            workflow = normalize_text(row.get("workflow"))
            process = normalize_text(row.get("process"))
            bucket = (workflow, process)
            engine = resolve_engine(args.engine, row)

            a_mb = parse_float(row.get("a_MB"), default=0.0)
            c_mb = parse_float(row.get("c_MB"), default=0.0)
            m_mb = parse_optional_float(row.get(HISTORICAL_MEMORY_COLUMN))
            predicted_memory_mb = parse_float(row.get(args.predicted_column))
            safe_memory_value_mb = parse_float(row.get(args.safe_column))
            planned_memory_value_mb = planned_memory_mb(
                predicted_memory_mb=predicted_memory_mb,
                safe_memory_mb=safe_memory_value_mb,
                mode=args.memory_for_plan,
            )

            novelty_feature = math.log1p(max(a_mb, 0.0))
            novelty_history = bucket_feature_history[bucket] if args.novelty_scope == "bucket" else global_feature_history
            novelty_score, novelty_distance = compute_novelty_score(
                feature_value=novelty_feature,
                history=novelty_history,
                novelty_scale=args.novelty_scale,
            )
            risk_score, historical_memory_used, log_cap_gap = compute_risk_score(
                residual_history=bucket_residual_history[bucket],
                predicted_memory_mb=predicted_memory_mb,
                allocation_mb=float(planned_memory_value_mb),
                safe_memory_mb=safe_memory_value_mb,
                cold_start_policy=args.cold_start_risk,
            )
            final_score = (
                args.weight_risk * risk_score
                + args.weight_novelty * novelty_score
            )
            selection_probability_pt = compute_selection_probability(
                final_score=final_score,
                tau_before=tau,
                temperature=args.selection_prob_temperature,
                epsilon=args.selection_prob_epsilon,
            )
            importance_weight = 1.0 / selection_probability_pt
            audit_flag = "Audit" if final_score > tau else "NoAudit"

            seen_count += 1
            if audit_flag == "Audit":
                audited_count += 1
            running_audit_rate = audited_count / seen_count

            pid_error = running_audit_rate - args.budget
            pid_integral = min(
                max(pid_integral + pid_error, args.pid_integral_min),
                args.pid_integral_max,
            )
            pid_derivative = pid_error - previous_error
            tau_after = min(
                max(
                    tau
                    + args.pid_kp * pid_error
                    + args.pid_ki * pid_integral
                    + args.pid_kd * pid_derivative,
                    args.tau_min,
                ),
                args.tau_max,
            )
            previous_error = pid_error

            task_type = build_task_type(row)
            task_instance_base, task_identity_source = resolve_task_instance_base(row, engine, row_index)
            task_instance = ensure_unique_task_instance(used_plan_keys, task_type, task_instance_base)

            scored_tasks.append(
                ScoredTask(
                    row_index=row_index,
                    workflow=workflow,
                    process=process,
                    engine=engine,
                    task_type=task_type,
                    task_instance=task_instance,
                    task_instance_base=task_instance_base,
                    task_identity_source=task_identity_source,
                    task_hash=normalize_text(row.get("task_hash")),
                    task_id=normalize_text(row.get("task_id")),
                    a_mb=a_mb,
                    c_mb=c_mb,
                    m_mb=m_mb,
                    predicted_memory_mb=predicted_memory_mb,
                    safe_memory_mb=safe_memory_value_mb,
                    planned_memory_mb=planned_memory_value_mb,
                    bucket_size_seen_before=len(bucket_feature_history[bucket]),
                    risk_history_size=len(bucket_residual_history[bucket]),
                    historical_memory_used=historical_memory_used,
                    novelty_feature_log_a=novelty_feature,
                    novelty_distance=novelty_distance,
                    novelty_score=novelty_score,
                    risk_score=risk_score,
                    log_cap_gap=log_cap_gap,
                    final_score=final_score,
                    selection_probability_pt=selection_probability_pt,
                    importance_weight_1_over_pt=importance_weight,
                    tau_before=tau,
                    tau_after=tau_after,
                    pid_error=pid_error,
                    running_audit_rate=running_audit_rate,
                    audit_flag=audit_flag,
                )
            )

            append_bounded_history(bucket_feature_history[bucket], novelty_feature, args.novelty_history_limit)
            append_bounded_history(global_feature_history, novelty_feature, args.novelty_history_limit)
            if m_mb is not None and predicted_memory_mb > 0:
                residual = math.log(max(m_mb, EPS)) - math.log(max(predicted_memory_mb, EPS))
                append_bounded_history(bucket_residual_history[bucket], residual, args.novelty_history_limit)

            tau = tau_after

    return scored_tasks, skipped_rows


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_scores_csv(path: str, scored_tasks: list[ScoredTask], prediction_model_name: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "row_index",
        "workflow",
        "process",
        "engine",
        "task_type",
        "task_instance",
        "task_instance_base",
        "task_identity_source",
        "task_hash",
        "task_id",
        "prediction_model",
        "a_MB",
        "c_MB",
        "M_MB",
        "predicted_memory_mb",
        "safe_memory_mb",
        "planned_memory_mb",
        "bucket_size_seen_before",
        "risk_history_size",
        "historical_memory_used",
        "novelty_feature_log_a",
        "novelty_distance",
        "novelty_score",
        "risk_score",
        "log_cap_gap",
        "final_score",
        "selection_probability_pt",
        "importance_weight_1_over_pt",
        "tau_before",
        "tau_after",
        "pid_error",
        "running_audit_rate",
        "audit_flag",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in scored_tasks:
            writer.writerow(
                {
                    "row_index": task.row_index,
                    "workflow": task.workflow,
                    "process": task.process,
                    "engine": task.engine,
                    "task_type": task.task_type,
                    "task_instance": task.task_instance,
                    "task_instance_base": task.task_instance_base,
                    "task_identity_source": task.task_identity_source,
                    "task_hash": task.task_hash,
                    "task_id": task.task_id,
                    "prediction_model": prediction_model_name,
                    "a_MB": f"{task.a_mb:.6f}",
                    "c_MB": f"{task.c_mb:.6f}",
                    "M_MB": format_optional_float(task.m_mb),
                    "predicted_memory_mb": f"{task.predicted_memory_mb:.6f}",
                    "safe_memory_mb": f"{task.safe_memory_mb:.6f}",
                    "planned_memory_mb": task.planned_memory_mb,
                    "bucket_size_seen_before": task.bucket_size_seen_before,
                    "risk_history_size": task.risk_history_size,
                    "historical_memory_used": str(task.historical_memory_used).lower(),
                    "novelty_feature_log_a": f"{task.novelty_feature_log_a:.6f}",
                    "novelty_distance": format_optional_float(task.novelty_distance),
                    "novelty_score": f"{task.novelty_score:.6f}",
                    "risk_score": f"{task.risk_score:.6f}",
                    "log_cap_gap": format_optional_float(task.log_cap_gap),
                    "final_score": f"{task.final_score:.6f}",
                    "selection_probability_pt": f"{task.selection_probability_pt:.6f}",
                    "importance_weight_1_over_pt": f"{task.importance_weight_1_over_pt:.6f}",
                    "tau_before": f"{task.tau_before:.6f}",
                    "tau_after": f"{task.tau_after:.6f}",
                    "pid_error": f"{task.pid_error:.6f}",
                    "running_audit_rate": f"{task.running_audit_rate:.6f}",
                    "audit_flag": task.audit_flag,
                }
            )


def write_task_plan_csv(path: str, scored_tasks: list[ScoredTask], prediction_model_name: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "task_type",
        "task_instance",
        "predicted_memory",
        "audit_flag",
        "workflow",
        "process",
        "engine",
        "task_identity_source",
        "prediction_model",
        "predicted_memory_mb",
        "safe_memory_mb",
        "planned_memory_mb",
        "task_hash",
        "task_id",
        "a_MB",
        "c_MB",
        "M_MB",
        "final_score",
        "risk_score",
        "novelty_score",
        "tau_before",
        "historical_memory_used",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in scored_tasks:
            writer.writerow(
                {
                    "task_type": task.task_type,
                    "task_instance": task.task_instance,
                    "predicted_memory": task.planned_memory_mb,
                    "audit_flag": task.audit_flag,
                    "workflow": task.workflow,
                    "process": task.process,
                    "engine": task.engine,
                    "task_identity_source": task.task_identity_source,
                    "prediction_model": prediction_model_name,
                    "predicted_memory_mb": f"{task.predicted_memory_mb:.6f}",
                    "safe_memory_mb": f"{task.safe_memory_mb:.6f}",
                    "planned_memory_mb": task.planned_memory_mb,
                    "task_hash": task.task_hash,
                    "task_id": task.task_id,
                    "a_MB": f"{task.a_mb:.6f}",
                    "c_MB": f"{task.c_mb:.6f}",
                    "M_MB": format_optional_float(task.m_mb),
                    "final_score": f"{task.final_score:.6f}",
                    "risk_score": f"{task.risk_score:.6f}",
                    "novelty_score": f"{task.novelty_score:.6f}",
                    "tau_before": f"{task.tau_before:.6f}",
                    "historical_memory_used": str(task.historical_memory_used).lower(),
                }
            )


def summarize(scored_tasks: list[ScoredTask], skipped_rows: int, prediction_model_name: str) -> None:
    if not scored_tasks:
        print("Scored rows: 0")
        print(f"Skipped rows: {skipped_rows}")
        return

    workflows = Counter(task.workflow for task in scored_tasks)
    engines = Counter(task.engine for task in scored_tasks)
    audits = sum(1 for task in scored_tasks if task.audit_flag == "Audit")
    with_actual_memory = sum(1 for task in scored_tasks if task.m_mb is not None)
    using_historical_memory = sum(1 for task in scored_tasks if task.historical_memory_used)
    avg_predicted = sum(task.predicted_memory_mb for task in scored_tasks) / len(scored_tasks)
    avg_safe = sum(task.safe_memory_mb for task in scored_tasks) / len(scored_tasks)

    print(f"Scored rows: {len(scored_tasks)}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Audit rows selected: {audits}")
    print(f"Observed audit rate: {audits / len(scored_tasks):.6f}")
    print(f"Prediction model: {prediction_model_name}")
    print(f"Average predicted peak memory (MB): {avg_predicted:.3f}")
    print(f"Average safe peak memory (MB): {avg_safe:.3f}")
    print(f"Rows carrying historical M_MB: {with_actual_memory}")
    print(f"Rows whose risk used historical residuals: {using_historical_memory}")
    print(f"Engines resolved: {dict(engines.most_common())}")
    print(f"Workflows covered: {dict(workflows.most_common(10))}")


def main() -> None:
    args = parse_args()
    if args.budget < 0 or args.budget > 1:
        raise SystemExit("--budget must be between 0 and 1")

    scored_tasks, skipped_rows = load_and_score_tasks(args)
    write_scores_csv(args.out_scores, scored_tasks, args.prediction_model_name)
    write_task_plan_csv(args.out_plan, scored_tasks, args.prediction_model_name)
    summarize(scored_tasks, skipped_rows, args.prediction_model_name)
    print(f"Scored CSV: {args.out_scores}")
    print(f"Task plan CSV: {args.out_plan}")


if __name__ == "__main__":
    main()
