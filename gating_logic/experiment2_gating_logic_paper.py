#!/usr/bin/env python3
"""Paper-style Experiment 2 gating logic for probabilistic memory models.

This script keeps the existing Experiment 2 implementation untouched and
implements the paper-oriented behavior instead:

1. Gate 1:
   audit when predictive variance is high.
2. Gate 2:
   audit when P(m(t) > Capacity_t) is high.
3. Gate 3:
   audit when the probabilistic prediction disagrees strongly with a cheap
   proxy model based on log(c), log(a), and log(c/a), where c is estimated
   online from historical averages of the same task type when needed.

The gates are evaluated in parallel. A task is marked Audit if ANY gate
exceeds the adaptive threshold tau_t. tau_t is updated online via a PID-style
controller to maintain the target global audit rate B.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass


HISTORICAL_MEMORY_COLUMN = "M_MB"
DEFAULT_MEAN_COLUMN = "pred_prob_joint_MB"
DEFAULT_SAFE_COLUMN = "safe_prob_joint_MB"
DEFAULT_STD_COLUMN = "std_prob_joint_MB"
DEFAULT_Q50_COLUMN = "q50_prob_joint_MB"
DEFAULT_Q95_COLUMN = "q95_prob_joint_MB"
DEFAULT_MODEL_NAME = "probabilistic_distribution_model"

ENGINE_TASK_INSTANCE_COLUMNS = {
    "nextflow": ("task_hash", "task_id", "native_id", "tag", "name", "sample", "hash"),
    "snakemake": ("task_id", "tag", "sample", "name", "native_id", "hash", "task_hash"),
    "pegasus": ("task_id", "name", "native_id", "tag", "sample", "hash", "task_hash"),
    "generic": ("task_id", "task_hash", "native_id", "tag", "name", "sample", "hash"),
}

ONE_SIDED_Z95 = 1.6448536269514722
SQRT2 = math.sqrt(2.0)


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
    c_hat_mb: float
    c_hat_source: str
    m_mb: float | None
    mean_memory_mb: float
    safe_memory_mb: float
    planned_memory_mb: int
    std_memory_mb: float | None
    q50_memory_mb: float | None
    q95_memory_mb: float | None
    predictive_variance_mb2: float
    gate1_score: float
    gate2_score: float
    gate3_score: float
    exceedance_probability: float
    proxy_pred_memory_mb: float | None
    proxy_residual_ratio: float | None
    distribution_source: str
    proxy_source: str
    max_gate_score: float
    selection_probability_pt: float
    importance_weight_1_over_pt: float
    tau_before: float
    tau_after: float
    pid_error: float
    running_audit_rate: float
    gate1_triggered: bool
    gate2_triggered: bool
    gate3_triggered: bool
    trigger_reason: str
    audit_flag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-style Experiment 2 gating with three parallel gates and an "
            "adaptive PID threshold."
        )
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV.")
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
        help="How task instances are resolved.",
    )
    parser.add_argument(
        "--prediction-model-name",
        default=DEFAULT_MODEL_NAME,
        help="Label written into output CSVs.",
    )
    parser.add_argument("--mean-column", default=DEFAULT_MEAN_COLUMN, help="Column containing predictive mean memory.")
    parser.add_argument("--safe-column", default=DEFAULT_SAFE_COLUMN, help="Column containing safe memory.")
    parser.add_argument("--std-column", default=DEFAULT_STD_COLUMN, help="Column containing predictive stddev memory.")
    parser.add_argument("--q50-column", default=DEFAULT_Q50_COLUMN, help="Optional median/50th percentile column.")
    parser.add_argument("--q95-column", default=DEFAULT_Q95_COLUMN, help="Optional 95th percentile column.")
    parser.add_argument(
        "--distribution",
        choices=("normal", "lognormal"),
        default="normal",
        help="Distribution family used to convert mean/std into exceedance probabilities.",
    )
    parser.add_argument(
        "--allow-safe-gap-fallback",
        action="store_true",
        help="If std/quantiles are missing, approximate spread from safe_memory - mean_memory.",
    )
    parser.add_argument(
        "--memory-for-plan",
        choices=("safe", "predicted", "max"),
        default="max",
        help="Memory written into the task plan.",
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
        "--variance-scale",
        type=float,
        default=0.25,
        help=(
            "Normalization scale for Gate 1 relative variance score, where "
            "relative_variance = (std/mean)^2."
        ),
    )
    parser.add_argument(
        "--proxy-pred-column",
        default="proxy_pred_M_MB",
        help="Optional precomputed proxy point-estimate memory column for Gate 3.",
    )
    parser.add_argument(
        "--proxy-disagreement-scale",
        type=float,
        default=0.50,
        help="Normalization scale for Gate 3 proxy-versus-probabilistic residual ratio.",
    )
    parser.add_argument(
        "--proxy-intercept",
        type=float,
        default=None,
        help="Proxy model intercept for const_1 + const_2*log(c) + const_3*log(a) + const_4*log(c/a).",
    )
    parser.add_argument("--proxy-log-c-coef", type=float, default=None, help="Proxy model coefficient for log(c).")
    parser.add_argument("--proxy-log-a-coef", type=float, default=None, help="Proxy model coefficient for log(a).")
    parser.add_argument(
        "--proxy-log-ca-ratio-coef",
        type=float,
        default=None,
        help="Proxy model coefficient for log(c/a).",
    )
    parser.add_argument(
        "--consumption-history-scope",
        choices=("task_type", "bucket", "global"),
        default="task_type",
        help="How to estimate predicted consumption c-hat for Gate 3.",
    )
    parser.add_argument(
        "--seed-current-consumption-if-cold",
        action="store_true",
        help="If no historical c-hat exists yet, fall back to the current row c_MB instead of 0.",
    )
    parser.add_argument(
        "--selection-prob-temperature",
        type=float,
        default=0.05,
        help="Temperature for the smooth audit-selection probability approximation p_t.",
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


def ensure_unique_task_instance(used_keys: dict[tuple[str, str], int], task_type: str, base_instance: str) -> str:
    key = (task_type, base_instance)
    count = used_keys.get(key, 0)
    used_keys[key] = count + 1
    if count == 0:
        return base_instance
    return f"{base_instance}__dup{count + 1}"


def planned_memory_mb(mean_memory_mb: float, safe_memory_mb: float, mode: str) -> int:
    if mode == "predicted":
        value = mean_memory_mb
    elif mode == "safe":
        value = safe_memory_mb
    else:
        value = max(mean_memory_mb, safe_memory_mb)
    return max(1, math.ceil(value))


def tail_probability_normal(mean_mb: float, std_mb: float, allocation_mb: float) -> float:
    if std_mb <= 0:
        return 1.0 if allocation_mb < mean_mb else 0.0
    z = (allocation_mb - mean_mb) / std_mb
    cdf = 0.5 * (1.0 + math.erf(z / SQRT2))
    return min(max(1.0 - cdf, 0.0), 1.0)


def tail_probability_lognormal(mean_mb: float, std_mb: float, allocation_mb: float) -> float:
    if allocation_mb <= 0:
        return 1.0
    if mean_mb <= 0 or std_mb <= 0:
        return 1.0 if allocation_mb < mean_mb else 0.0
    sigma2_log = math.log(1.0 + (std_mb * std_mb) / (mean_mb * mean_mb + 1e-12))
    sigma_log = math.sqrt(max(sigma2_log, 1e-12))
    mu_log = math.log(mean_mb) - 0.5 * sigma2_log
    z = (math.log(allocation_mb) - mu_log) / sigma_log
    cdf = 0.5 * (1.0 + math.erf(z / SQRT2))
    return min(max(1.0 - cdf, 0.0), 1.0)


def infer_distribution_params(
    mean_memory_mb: float,
    safe_memory_mb: float,
    std_memory_mb: float | None,
    q50_memory_mb: float | None,
    q95_memory_mb: float | None,
    allow_safe_gap_fallback: bool,
) -> tuple[float | None, str]:
    if std_memory_mb is not None and std_memory_mb > 0:
        return std_memory_mb, "std_column"
    if q50_memory_mb is not None and q95_memory_mb is not None and q95_memory_mb > q50_memory_mb:
        return (q95_memory_mb - q50_memory_mb) / ONE_SIDED_Z95, "q50_q95_width"
    if allow_safe_gap_fallback and safe_memory_mb > mean_memory_mb:
        return safe_memory_mb - mean_memory_mb, "safe_gap_proxy"
    return None, "missing"


def compute_gate1_variance_score(mean_memory_mb: float, std_memory_mb: float | None, variance_scale: float) -> tuple[float, float]:
    if std_memory_mb is None or std_memory_mb <= 0 or mean_memory_mb <= 0:
        return 0.0, 0.0
    predictive_variance_mb2 = std_memory_mb * std_memory_mb
    relative_variance = (std_memory_mb / max(mean_memory_mb, 1e-6)) ** 2
    score = min(relative_variance / max(variance_scale, 1e-6), 1.0)
    return score, predictive_variance_mb2


def compute_gate2_risk_score(mean_memory_mb: float, std_memory_mb: float | None, allocation_mb: float, distribution: str) -> float:
    if std_memory_mb is None:
        return 0.5
    if distribution == "lognormal":
        return tail_probability_lognormal(mean_memory_mb, std_memory_mb, allocation_mb)
    return tail_probability_normal(mean_memory_mb, std_memory_mb, allocation_mb)


def compute_proxy_prediction_from_formula(
    a_mb: float,
    c_hat_mb: float,
    intercept: float | None,
    log_c_coef: float | None,
    log_a_coef: float | None,
    log_ca_ratio_coef: float | None,
) -> float | None:
    if None in (intercept, log_c_coef, log_a_coef, log_ca_ratio_coef):
        return None
    if a_mb <= 0 or c_hat_mb <= 0:
        return None
    log_c = math.log(c_hat_mb)
    log_a = math.log(a_mb)
    log_ca_ratio = math.log(c_hat_mb / a_mb)
    return (
        intercept
        + log_c_coef * log_c
        + log_a_coef * log_a
        + log_ca_ratio_coef * log_ca_ratio
    )


def compute_proxy_disagreement_score(mean_memory_mb: float, proxy_pred_memory_mb: float, disagreement_scale: float) -> tuple[float, float]:
    baseline = max(abs(mean_memory_mb), abs(proxy_pred_memory_mb), 1e-6)
    disagreement_ratio = abs(mean_memory_mb - proxy_pred_memory_mb) / baseline
    score = min(disagreement_ratio / max(disagreement_scale, 1e-6), 1.0)
    return score, disagreement_ratio


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def compute_selection_probability(max_gate_score: float, tau_before: float, temperature: float, epsilon: float) -> float:
    temp = max(temperature, 1e-9)
    margin = (max_gate_score - tau_before) / temp
    if margin >= 0:
        z = math.exp(-margin)
        p = 1.0 / (1.0 + z)
    else:
        z = math.exp(margin)
        p = z / (1.0 + z)
    return clamp(p, epsilon, 1.0)


def should_keep_row(row: dict[str, str], workflows: set[str], mean_column: str, safe_column: str) -> bool:
    if workflows and normalize_text(row.get("workflow")) not in workflows:
        return False
    return bool(normalize_text(row.get(mean_column)) and normalize_text(row.get(safe_column)))


def c_history_key(scope: str, workflow: str, process: str, task_type: str) -> str | tuple[str, str]:
    if scope == "global":
        return "__global__"
    if scope == "bucket":
        return (workflow, process)
    return task_type


def estimate_consumption(
    scope: str,
    workflow: str,
    process: str,
    task_type: str,
    c_mb: float,
    historical_sum: dict[object, float],
    historical_count: dict[object, int],
    global_sum: float,
    global_count: int,
    seed_current_if_cold: bool,
) -> tuple[float, str]:
    history_key = c_history_key(scope, workflow, process, task_type)
    if historical_count.get(history_key, 0) > 0:
        return historical_sum[history_key] / historical_count[history_key], f"{scope}_historical_average"
    if global_count > 0:
        return global_sum / global_count, "global_historical_average"
    if seed_current_if_cold:
        return c_mb, "current_row_fallback"
    return 0.0, "cold_start_zero"


def load_and_score_tasks(args: argparse.Namespace) -> tuple[list[ScoredTask], int]:
    workflows = set(args.workflow)
    used_plan_keys: dict[tuple[str, str], int] = {}
    skipped_rows = 0
    scored_tasks: list[ScoredTask] = []

    historical_c_sum: dict[object, float] = defaultdict(float)
    historical_c_count: dict[object, int] = defaultdict(int)
    global_c_sum = 0.0
    global_c_count = 0

    tau = clamp(args.tau_init, args.tau_min, args.tau_max)
    pid_integral = 0.0
    previous_error = 0.0
    audited_count = 0
    seen_count = 0

    with open(args.predictions, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            if not should_keep_row(row, workflows, args.mean_column, args.safe_column):
                skipped_rows += 1
                continue

            workflow = normalize_text(row.get("workflow"))
            process = normalize_text(row.get("process"))
            engine = resolve_engine(args.engine, row)
            task_type = build_task_type(row)

            a_mb = parse_float(row.get("a_MB"), default=0.0)
            c_mb = parse_float(row.get("c_MB"), default=0.0)
            m_mb = parse_optional_float(row.get(HISTORICAL_MEMORY_COLUMN))
            mean_memory_mb = parse_float(row.get(args.mean_column))
            safe_memory_value_mb = parse_float(row.get(args.safe_column))
            q50_memory_mb = parse_optional_float(row.get(args.q50_column))
            q95_memory_mb = parse_optional_float(row.get(args.q95_column))
            inferred_std_mb, distribution_source = infer_distribution_params(
                mean_memory_mb=mean_memory_mb,
                safe_memory_mb=safe_memory_value_mb,
                std_memory_mb=parse_optional_float(row.get(args.std_column)),
                q50_memory_mb=q50_memory_mb,
                q95_memory_mb=q95_memory_mb,
                allow_safe_gap_fallback=args.allow_safe_gap_fallback,
            )
            planned_memory_value_mb = planned_memory_mb(
                mean_memory_mb=mean_memory_mb,
                safe_memory_mb=safe_memory_value_mb,
                mode=args.memory_for_plan,
            )

            gate1_score, predictive_variance_mb2 = compute_gate1_variance_score(
                mean_memory_mb=mean_memory_mb,
                std_memory_mb=inferred_std_mb,
                variance_scale=args.variance_scale,
            )
            gate2_score = compute_gate2_risk_score(
                mean_memory_mb=mean_memory_mb,
                std_memory_mb=inferred_std_mb,
                allocation_mb=float(planned_memory_value_mb),
                distribution=args.distribution,
            )

            c_hat_mb, c_hat_source = estimate_consumption(
                scope=args.consumption_history_scope,
                workflow=workflow,
                process=process,
                task_type=task_type,
                c_mb=c_mb,
                historical_sum=historical_c_sum,
                historical_count=historical_c_count,
                global_sum=global_c_sum,
                global_count=global_c_count,
                seed_current_if_cold=args.seed_current_consumption_if_cold,
            )

            proxy_pred_memory_mb = parse_optional_float(row.get(args.proxy_pred_column))
            if proxy_pred_memory_mb is not None:
                gate3_score, proxy_residual_ratio = compute_proxy_disagreement_score(
                    mean_memory_mb=mean_memory_mb,
                    proxy_pred_memory_mb=proxy_pred_memory_mb,
                    disagreement_scale=args.proxy_disagreement_scale,
                )
                proxy_source = "proxy_prediction_column"
            else:
                proxy_pred_memory_mb = compute_proxy_prediction_from_formula(
                    a_mb=a_mb,
                    c_hat_mb=c_hat_mb,
                    intercept=args.proxy_intercept,
                    log_c_coef=args.proxy_log_c_coef,
                    log_a_coef=args.proxy_log_a_coef,
                    log_ca_ratio_coef=args.proxy_log_ca_ratio_coef,
                )
                if proxy_pred_memory_mb is not None:
                    gate3_score, proxy_residual_ratio = compute_proxy_disagreement_score(
                        mean_memory_mb=mean_memory_mb,
                        proxy_pred_memory_mb=proxy_pred_memory_mb,
                        disagreement_scale=args.proxy_disagreement_scale,
                    )
                    proxy_source = "proxy_formula_from_c_hat"
                else:
                    gate3_score = 0.0
                    proxy_residual_ratio = None
                    proxy_source = "unavailable"

            gate1_triggered = gate1_score > tau
            gate2_triggered = gate2_score > tau
            gate3_triggered = gate3_score > tau
            max_gate_score = max(gate1_score, gate2_score, gate3_score)
            selection_probability_pt = compute_selection_probability(
                max_gate_score=max_gate_score,
                tau_before=tau,
                temperature=args.selection_prob_temperature,
                epsilon=args.selection_prob_epsilon,
            )
            importance_weight = 1.0 / selection_probability_pt
            audit_flag = "Audit" if (gate1_triggered or gate2_triggered or gate3_triggered) else "NoAudit"

            trigger_parts = []
            if gate1_triggered:
                trigger_parts.append("gate1_variance")
            if gate2_triggered:
                trigger_parts.append("gate2_failure_probability")
            if gate3_triggered:
                trigger_parts.append("gate3_proxy_residual")
            trigger_reason = ",".join(trigger_parts) if trigger_parts else "none"

            seen_count += 1
            if audit_flag == "Audit":
                audited_count += 1
            running_audit_rate = audited_count / seen_count

            pid_error = running_audit_rate - args.budget
            pid_integral = clamp(
                pid_integral + pid_error,
                args.pid_integral_min,
                args.pid_integral_max,
            )
            pid_derivative = pid_error - previous_error
            tau_after = clamp(
                tau
                + args.pid_kp * pid_error
                + args.pid_ki * pid_integral
                + args.pid_kd * pid_derivative,
                args.tau_min,
                args.tau_max,
            )
            previous_error = pid_error

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
                    c_hat_mb=c_hat_mb,
                    c_hat_source=c_hat_source,
                    m_mb=m_mb,
                    mean_memory_mb=mean_memory_mb,
                    safe_memory_mb=safe_memory_value_mb,
                    planned_memory_mb=planned_memory_value_mb,
                    std_memory_mb=inferred_std_mb,
                    q50_memory_mb=q50_memory_mb,
                    q95_memory_mb=q95_memory_mb,
                    predictive_variance_mb2=predictive_variance_mb2,
                    gate1_score=gate1_score,
                    gate2_score=gate2_score,
                    gate3_score=gate3_score,
                    exceedance_probability=gate2_score,
                    proxy_pred_memory_mb=proxy_pred_memory_mb,
                    proxy_residual_ratio=proxy_residual_ratio,
                    distribution_source=distribution_source,
                    proxy_source=proxy_source,
                    max_gate_score=max_gate_score,
                    selection_probability_pt=selection_probability_pt,
                    importance_weight_1_over_pt=importance_weight,
                    tau_before=tau,
                    tau_after=tau_after,
                    pid_error=pid_error,
                    running_audit_rate=running_audit_rate,
                    gate1_triggered=gate1_triggered,
                    gate2_triggered=gate2_triggered,
                    gate3_triggered=gate3_triggered,
                    trigger_reason=trigger_reason,
                    audit_flag=audit_flag,
                )
            )

            tau = tau_after

            history_key = c_history_key(args.consumption_history_scope, workflow, process, task_type)
            historical_c_sum[history_key] += c_mb
            historical_c_count[history_key] += 1
            global_c_sum += c_mb
            global_c_count += 1

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
        "c_hat_MB",
        "c_hat_source",
        "M_MB",
        "mean_memory_mb",
        "std_memory_mb",
        "q50_memory_mb",
        "q95_memory_mb",
        "safe_memory_mb",
        "planned_memory_mb",
        "predictive_variance_mb2",
        "gate1_score",
        "gate2_score",
        "gate3_score",
        "exceedance_probability",
        "proxy_pred_memory_mb",
        "proxy_residual_ratio",
        "distribution_source",
        "proxy_source",
        "max_gate_score",
        "selection_probability_pt",
        "importance_weight_1_over_pt",
        "tau_before",
        "tau_after",
        "pid_error",
        "running_audit_rate",
        "gate1_triggered",
        "gate2_triggered",
        "gate3_triggered",
        "trigger_reason",
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
                    "c_hat_MB": f"{task.c_hat_mb:.6f}",
                    "c_hat_source": task.c_hat_source,
                    "M_MB": format_optional_float(task.m_mb),
                    "mean_memory_mb": f"{task.mean_memory_mb:.6f}",
                    "std_memory_mb": format_optional_float(task.std_memory_mb),
                    "q50_memory_mb": format_optional_float(task.q50_memory_mb),
                    "q95_memory_mb": format_optional_float(task.q95_memory_mb),
                    "safe_memory_mb": f"{task.safe_memory_mb:.6f}",
                    "planned_memory_mb": task.planned_memory_mb,
                    "predictive_variance_mb2": f"{task.predictive_variance_mb2:.6f}",
                    "gate1_score": f"{task.gate1_score:.6f}",
                    "gate2_score": f"{task.gate2_score:.6f}",
                    "gate3_score": f"{task.gate3_score:.6f}",
                    "exceedance_probability": f"{task.exceedance_probability:.6f}",
                    "proxy_pred_memory_mb": format_optional_float(task.proxy_pred_memory_mb),
                    "proxy_residual_ratio": format_optional_float(task.proxy_residual_ratio),
                    "distribution_source": task.distribution_source,
                    "proxy_source": task.proxy_source,
                    "max_gate_score": f"{task.max_gate_score:.6f}",
                    "selection_probability_pt": f"{task.selection_probability_pt:.6f}",
                    "importance_weight_1_over_pt": f"{task.importance_weight_1_over_pt:.6f}",
                    "tau_before": f"{task.tau_before:.6f}",
                    "tau_after": f"{task.tau_after:.6f}",
                    "pid_error": f"{task.pid_error:.6f}",
                    "running_audit_rate": f"{task.running_audit_rate:.6f}",
                    "gate1_triggered": str(task.gate1_triggered),
                    "gate2_triggered": str(task.gate2_triggered),
                    "gate3_triggered": str(task.gate3_triggered),
                    "trigger_reason": task.trigger_reason,
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
        "mean_memory_mb",
        "std_memory_mb",
        "q50_memory_mb",
        "q95_memory_mb",
        "safe_memory_mb",
        "planned_memory_mb",
        "c_hat_MB",
        "c_hat_source",
        "proxy_pred_memory_mb",
        "proxy_residual_ratio",
        "max_gate_score",
        "selection_probability_pt",
        "importance_weight_1_over_pt",
        "task_hash",
        "task_id",
        "a_MB",
        "c_MB",
        "M_MB",
        "exceedance_probability",
        "gate1_score",
        "gate2_score",
        "gate3_score",
        "tau_before",
        "trigger_reason",
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
                    "mean_memory_mb": f"{task.mean_memory_mb:.6f}",
                    "std_memory_mb": format_optional_float(task.std_memory_mb),
                    "q50_memory_mb": format_optional_float(task.q50_memory_mb),
                    "q95_memory_mb": format_optional_float(task.q95_memory_mb),
                    "safe_memory_mb": f"{task.safe_memory_mb:.6f}",
                    "planned_memory_mb": task.planned_memory_mb,
                    "c_hat_MB": f"{task.c_hat_mb:.6f}",
                    "c_hat_source": task.c_hat_source,
                    "proxy_pred_memory_mb": format_optional_float(task.proxy_pred_memory_mb),
                    "proxy_residual_ratio": format_optional_float(task.proxy_residual_ratio),
                    "max_gate_score": f"{task.max_gate_score:.6f}",
                    "selection_probability_pt": f"{task.selection_probability_pt:.6f}",
                    "importance_weight_1_over_pt": f"{task.importance_weight_1_over_pt:.6f}",
                    "task_hash": task.task_hash,
                    "task_id": task.task_id,
                    "a_MB": f"{task.a_mb:.6f}",
                    "c_MB": f"{task.c_mb:.6f}",
                    "M_MB": format_optional_float(task.m_mb),
                    "exceedance_probability": f"{task.exceedance_probability:.6f}",
                    "gate1_score": f"{task.gate1_score:.6f}",
                    "gate2_score": f"{task.gate2_score:.6f}",
                    "gate3_score": f"{task.gate3_score:.6f}",
                    "tau_before": f"{task.tau_before:.6f}",
                    "trigger_reason": task.trigger_reason,
                }
            )


def summarize(scored_tasks: list[ScoredTask], skipped_rows: int, prediction_model_name: str) -> None:
    if not scored_tasks:
        print("Scored rows: 0")
        print(f"Skipped rows: {skipped_rows}")
        return

    audits = sum(1 for task in scored_tasks if task.audit_flag == "Audit")
    workflows = Counter(task.workflow for task in scored_tasks)
    engines = Counter(task.engine for task in scored_tasks)
    trigger_counts = Counter(task.trigger_reason for task in scored_tasks if task.trigger_reason != "none")
    avg_tau = sum(task.tau_before for task in scored_tasks) / len(scored_tasks)
    final_tau = scored_tasks[-1].tau_after
    avg_pt = sum(task.selection_probability_pt for task in scored_tasks) / len(scored_tasks)

    print(f"Scored rows: {len(scored_tasks)}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Audit rows selected: {audits}")
    print(f"Observed audit rate: {audits / len(scored_tasks):.6f}")
    print(f"Average selection probability p_t: {avg_pt:.6f}")
    print(f"Average tau: {avg_tau:.6f}")
    print(f"Final tau: {final_tau:.6f}")
    print(f"Prediction model: {prediction_model_name}")
    print(f"Engines resolved: {dict(engines.most_common())}")
    print(f"Workflows covered: {dict(workflows.most_common(10))}")
    print(f"Trigger counts: {dict(trigger_counts.most_common())}")


def main() -> None:
    args = parse_args()
    if args.budget < 0 or args.budget > 1:
        raise SystemExit("--budget must be between 0 and 1")
    if args.tau_min > args.tau_max:
        raise SystemExit("--tau-min cannot exceed --tau-max")

    scored_tasks, skipped_rows = load_and_score_tasks(args)
    write_scores_csv(args.out_scores, scored_tasks, args.prediction_model_name)
    write_task_plan_csv(args.out_plan, scored_tasks, args.prediction_model_name)
    summarize(scored_tasks, skipped_rows, args.prediction_model_name)
    print(f"Scored CSV: {args.out_scores}")
    print(f"Task plan CSV: {args.out_plan}")


if __name__ == "__main__":
    main()
