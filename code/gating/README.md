# Gating Logic

This directory contains the paper-faithful gating entry points used between the baseline prediction stage and the selective audit rerun stage.

Sequence:

1. Run the baseline workflow first, as described in [workflows/README.md](../../workflows/README.md).
2. Generate or collect the prediction CSV for that experiment.
3. Run the appropriate gating command below.
4. Use the generated `task_plan.csv` in the selective `eBPF` rerun command for Experiment 2, or use it to filter `per_task_after_rerun.csv` before the replay-based Experiment 1 `strace` replay step.

The two scripts in this directory always write:

- one detailed `scores` CSV for inspection
- one `task_plan.csv` file that contains `task_type`, `task_instance`, `predicted_memory`, and `audit_flag`

`audit_flag=Audit` means that task instance should receive dynamic auditing in the next rerun. `audit_flag=NoAudit` means the task should run normally without dynamic auditing.

## Inputs

Experiment 1 uses point-prediction outputs such as:

- [results/predictions_all.csv](../../results/predictions_all.csv)

Experiment 2 uses probabilistic prediction outputs such as:

- [results/predictions_qlgbm_exp2_all.csv](../../results/predictions_qlgbm_exp2_all.csv)

If you are reproducing only one workflow, you may add `--workflow <workflow_name>` to keep only that workflow's rows.

## Experiment 1

Use [experiment1_gating_logic_paper.py](experiment1_gating_logic_paper.py) for the paper-faithful Section IV-A point-prediction gating path.

Example command:

```bash
cd IEEE_CLUSTER_2026_CAMP_ARTIFACT

python code/gating/experiment1_gating_logic_paper.py \
  --predictions results/predictions_all.csv \
  --prediction-model-name lgbm_sizey \
  --predicted-column pred_lgbm_sizey_MB \
  --safe-column safe_lgbm_sizey_MB \
  --budget 0.10 \
  --out-scores code/gating/exp1_lgbm_sizey_scores_b_0.1.csv \
  --out-plan code/gating/exp1_lgbm_sizey_task_plan_b_0.1.csv
```

Workflow-specific example:

```bash
cd IEEE_CLUSTER_2026_CAMP_ARTIFACT

python code/gating/experiment1_gating_logic_paper.py \
  --predictions results/predictions_all.csv \
  --workflow eager \
  --prediction-model-name lgbm_sizey \
  --predicted-column pred_lgbm_sizey_MB \
  --safe-column safe_lgbm_sizey_MB \
  --budget 0.10 \
  --out-scores code/gating/eager_exp1_scores.csv \
  --out-plan code/gating/eager_exp1_task_plan.csv
```

This produces:

- `code/gating/exp1_lgbm_sizey_scores_b_0.1.csv`
- `code/gating/exp1_lgbm_sizey_task_plan_b_0.1.csv`

Use the generated `task_plan.csv` to select the `Audit` subset from `per_task_after_rerun.csv` before running the replay-based Experiment 1 `strace` replay step. The shared filter command is:

```bash
python audit/strace/filter_after_by_task_plan.py \
  --after workflows/<workflow_name>/per_task_after_rerun.csv \
  --task-plan code/gating/<workflow_name>_exp1_task_plan.csv \
  --output workflows/<workflow_name>/per_task_after_audit.csv
```

The filtered `per_task_after_audit.csv` is then the input to the workflow-specific `replay_c_bytes_from_after_csv.sh` command described in [workflows/README.md](../../workflows/README.md).

## Experiment 2

Use [experiment2_gating_logic_paper.py](experiment2_gating_logic_paper.py) for the paper-faithful probabilistic gating path.

Example command:

```bash
cd IEEE_CLUSTER_2026_CAMP_ARTIFACT

python code/gating/experiment2_gating_logic_paper.py \
  --predictions results/predictions_qlgbm_exp2_all.csv \
  --prediction-model-name qlgbm_sizey \
  --mean-column pred_qlgbm_sizey_MB \
  --safe-column safe_qlgbm_sizey_MB \
  --std-column std_qlgbm_sizey_MB \
  --q50-column q50_qlgbm_sizey_MB \
  --q95-column q95_qlgbm_sizey_MB \
  --budget 0.10 \
  --out-scores code/gating/exp2_qlgbm_sizey_scores_b_0.1.csv \
  --out-plan code/gating/exp2_qlgbm_sizey_task_plan_b_0.1.csv
```

Workflow-specific examples:

```bash
cd IEEE_CLUSTER_2026_CAMP_ARTIFACT

python code/gating/experiment2_gating_logic_paper.py \
  --predictions results/predictions_qlgbm_exp2_all.csv \
  --workflow bowtie2_audit_nf \
  --prediction-model-name qlgbm_sizey \
  --mean-column pred_qlgbm_sizey_MB \
  --safe-column safe_qlgbm_sizey_MB \
  --std-column std_qlgbm_sizey_MB \
  --q50-column q50_qlgbm_sizey_MB \
  --q95-column q95_qlgbm_sizey_MB \
  --budget 0.10 \
  --out-scores code/gating/bowtie_exp2_scores.csv \
  --out-plan code/gating/bowtie_exp2_task_plan.csv

python code/gating/experiment2_gating_logic_paper.py \
  --predictions results/predictions_qlgbm_exp2_all.csv \
  --workflow minimap2_audit_nf \
  --prediction-model-name qlgbm_sizey \
  --mean-column pred_qlgbm_sizey_MB \
  --safe-column safe_qlgbm_sizey_MB \
  --std-column std_qlgbm_sizey_MB \
  --q50-column q50_qlgbm_sizey_MB \
  --q95-column q95_qlgbm_sizey_MB \
  --budget 0.10 \
  --out-scores code/gating/minimap_exp2_scores.csv \
  --out-plan code/gating/minimap_exp2_task_plan.csv

python code/gating/experiment2_gating_logic_paper.py \
  --predictions results/predictions_qlgbm_exp2_all.csv \
  --workflow mcmicro \
  --prediction-model-name qlgbm_sizey \
  --mean-column pred_qlgbm_sizey_MB \
  --safe-column safe_qlgbm_sizey_MB \
  --std-column std_qlgbm_sizey_MB \
  --q50-column q50_qlgbm_sizey_MB \
  --q95-column q95_qlgbm_sizey_MB \
  --budget 0.10 \
  --out-scores code/gating/mcmicro_exp2_scores.csv \
  --out-plan code/gating/mcmicro_exp2_task_plan.csv
```

This produces:

- `code/gating/exp2_qlgbm_sizey_scores_b_0.1.csv`
- `code/gating/exp2_qlgbm_sizey_task_plan_b_0.1.csv`

Use the generated `task_plan.csv` in the selective eBPF rerun command for `bowtie`, `minimap`, or `mcmicro` from [workflows/README.md](../../workflows/README.md). The selective eBPF wrapper reads `audit_flag` and attaches eBPF only for rows marked `Audit`.

## Output To Workflow Handoff

After you generate the task plan, pass it into the workflow rerun command as `<task_plan.csv>`.

For Experiment 2, the workflow README already shows the required parameter:

- `--camp_task_plan <task_plan.csv>`

For Experiment 1, pass the generated task plan into `audit/strace/filter_after_by_task_plan.py`, then use the filtered `per_task_after_audit.csv` in the workflow-specific `strace` replay step.
