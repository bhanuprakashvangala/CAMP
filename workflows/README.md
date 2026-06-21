# Workflows

This README accompanies the workflow-level artifact for the IEEE Cluster 2026 CAMP paper. It is written for artifact review and collects, in one place, the dataset provenance, workflow source location, execution sequence, and compact CSV outputs needed to reproduce and compare the preserved workflow runs.

Use the paper-level compute environment as the common reproduction target unless a workflow section states otherwise. In the paper, the experiments were run on a `20-node` `Slurm` cluster on `AWS ParallelCluster`, using `r5.8xlarge` compute nodes with `32 vCPU` and `256 GB RAM` per node, `EFS` shared storage, and `Apptainer` containers.

The workflows below assume the environment provides `Nextflow`, `Java`, `Python 3`, `Slurm`, `Apptainer` or `Singularity`, and `strace` on the compute nodes. The post-run compact CSV reconstruction uses the shared strace helper scripts in [../audit/strace](../audit/strace).

Command structure used throughout this README:
- Replay-based Experiment 1 workflows: first run the workflow normally to generate the workflow outputs, `nf-trace-rerun.txt`, and `per_task_after_rerun.csv`; then filter `per_task_after_rerun.csv` by the Experiment 1 `task_plan.csv` and run the bundled `strace` replay and merge steps on only the `Audit` subset
- Experiment 2 workflows: first run the workflow normally without eBPF; then apply the gating logic and rerun only the task instances marked `Audit` with the eBPF overlay enabled

The shared artifact-level eBPF notes and helper scripts are collected in [../audit/ebpf/README.md](../audit/ebpf/README.md).
The exact gating commands that generate `scores` CSVs and `task_plan.csv` files are collected in [../code/gating/README.md](../code/gating/README.md).

## Workflow To ML Handoff

After an individual workflow rerun, the next step for the CAMP artifact is memory-prediction ingestion. The first prediction stage uses task tables that provide at least `workflow`, `process`, and `a_bytes`. For the replay-based Experiment 1 workflows, the bundled scripts can filter `per_task_after_rerun.csv` by the generated `task_plan.csv` and then reconstruct `strace`-derived dynamic features only for the `Audit` subset. For Experiment 2, the selective eBPF wrapper collects dynamic features only for the task instances marked `Audit`. In both cases, the resulting `c_bytes` values can then be used for the enhanced joint-view prediction stage.

For `eager`, `rnaseq`, `mag`, `mag_karlsson`, and `chipseq`, the workflow sections below produce the rerun trace and reconstruction files needed to assemble the per-task ML input. For the point-based memory-prediction commands themselves, follow [../README.md](../README.md), especially [code/reruns/train_full.py](../code/reruns/train_full.py) and [code/reruns/predict_memory_unified.py](../code/reruns/predict_memory_unified.py).

## Methylseq

Dataset download info: wget https://ftp.ensembl.org/pub/release-104/fasta/rattus_norvegicus/dna/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz

Use [samplesheet_traced_remote.csv](methylseq/samplesheet_traced_remote.csv) as the exact `640`-sample remote manifest matching the recorded methylseq strace population. It covers all traced sample tags that appear in the recorded aggregated task list.

Reproducing the preserved methylseq inputs requires `Nextflow` with `Java`, a Conda-compatible setup with `mamba` or `micromamba`, `strace` on the execution nodes, and a `Slurm` environment (the recorded run used Slurm-backed configs).

Clone the upstream workflow source from `https://github.com/nf-core/methylseq.git` and check out the recorded local snapshot revision `5aa56467a85a5e2d6795ea72dfa5a5f0c9babc23` before running the commands below. The workflow entrypoint in the cloned repository is `main.nf`.

Recorded compact comparison CSVs included in [methylseq](methylseq):
- [task_instances_traced.csv](methylseq/task_instances_traced.csv) as the recorded compact `process,tag,status` task-instance list with `2350` data rows
- [task_instance_summary_traced.csv](methylseq/task_instance_summary_traced.csv) as the recorded per-process task-instance counts

```bash
# 1. clone the workflow source and pin it to the recorded snapshot
git clone https://github.com/nf-core/methylseq.git
cd methylseq
git checkout 5aa56467a85a5e2d6795ea72dfa5a5f0c9babc23

# 2. download the exact traced FASTQ population used by the recorded task list
mkdir -p <fastq_data_dir>
wget -c -i ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/fastq_urls_traced.txt -P <fastq_data_dir>

# 3. download the recorded reference FASTA
mkdir -p <reference_dir>
wget -c https://ftp.ensembl.org/pub/release-104/fasta/rattus_norvegicus/dna/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz -P <reference_dir>

# 4. rewrite the traced remote samplesheet so the FASTQ paths point to your local dataset directory
python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/scripts/rewrite_samplesheet_paths.py \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/samplesheet_traced_remote.csv \
  --data-dir <fastq_data_dir> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/samplesheet_traced_local.csv

# 5. baseline workflow rerun without strace
nextflow run . -profile mamba \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/configs/slurm_mamba_nostrace.config -qs 128 \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/samplesheet_traced_local.csv \
  --outdir <baseline_results_dir> \
  -work-dir <baseline_work_dir> \
  --igenomes_ignore \
  --fasta <reference_dir>/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz \
  --save_reference

# 6. instrumented strace rerun for dynamic features
nextflow run . -profile mamba \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/configs/slurm_mamba.config -qs 128 \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/samplesheet_traced_local.csv \
  --outdir <results_dir> \
  -work-dir <work_dir> \
  --igenomes_ignore \
  --fasta <reference_dir>/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz \
  --save_reference

# 7. resumed balanced strace rerun, if you need to continue the recorded retry path
nextflow run . -resume -profile mamba \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/configs/slurm_mamba_balanced.config \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/methylseq/samplesheet_traced_local.csv \
  --outdir <results_dir> \
  -work-dir <work_dir> \
  --igenomes_ignore \
  --fasta <reference_dir>/Rattus_norvegicus.Rnor_6.0.dna.toplevel.fa.gz \
  --save_reference
```

Command order for `methylseq`:
- use the baseline no-`strace` rerun to verify the workflow runs successfully on the preserved `640`-sample manifest
- use the instrumented `strace` rerun to compare task-instance counts against [task_instances_traced.csv](methylseq/task_instances_traced.csv) and [task_instance_summary_traced.csv](methylseq/task_instance_summary_traced.csv)

The recorded `2350` task rows include `COMPLETED`, `CACHED`, and `ABORTED` entries from the paper's run history, so the exact status mix can differ if the rerun does not follow the same initial plus resume path.

## Eager

The [eager](eager) folder holds the preserved nf-core/eager run. The preserved compact run in this repo produces `36` task rows across `16` task types. The raw `nf-trace` snapshot has `41` rows because it also preserves retries and failed attempts.

The local `eager/` checkout in this repo is not a complete runnable pipeline snapshot, so clone the upstream workflow source instead. The preserved run used nf-core/eager `v2.5.3` from `https://github.com/nf-core/eager.git`.

Command order for `eager`:
- baseline rerun: steps `1` through `5`
- filter the rerun task table by the Experiment 1 `task_plan.csv`, then run the `strace`-based dynamic-feature reconstruction in steps `6` through `8`

```bash
# 1. clone the workflow source used by the preserved run
git clone https://github.com/nf-core/eager.git
cd eager
git checkout 2.5.3

# 2. download the run-sheet FASTQs used by the preserved compact output
#    use the three accessions in samplesheet_remote.tsv: ERR2564026, ERR2564029, ERR2564033
#    place the resulting FASTQ files in <eager_fastq_dir>

# 3. provide the GRCh38 FASTA used by the preserved run
#    place Homo_sapiens_assembly38.fasta in <reference_dir>

# 4. rewrite the preserved samplesheet so it points to your local FASTQ directory
python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/scripts/rewrite_samplesheet_paths.py \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/samplesheet_remote.tsv \
  --data-dir <eager_fastq_dir> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/samplesheet_local.tsv

# 5. rerun the workflow and preserve the rerun trace / after-task CSVs in the artifact folder
nextflow run . -profile singularity \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/configs/slurm_repro.config \
  --camp_output_dir ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager \
  --outdir <results_dir> \
  -work-dir <work_dir> \
  --igenomes_ignore \
  --igenomes_base <dummy_igenomes_dir> \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/samplesheet_local.tsv \
  --fasta <reference_dir>/Homo_sapiens_assembly38.fasta

# 6. keep only the Audit rows from the Experiment 1 task plan
python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/filter_after_by_task_plan.py \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/per_task_after_rerun.csv \
  --task-plan <task_plan.csv> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/per_task_after_audit.csv

# 7. replay only the audited rerun work directories under strace to produce c-bytes
bash ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/replay_c_bytes_from_after_csv.sh \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/per_task_after_audit.csv \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/per_task_c_rerun.csv

# 8. build the compact rerun CSV and compare it against task_metrics_recorded.csv
python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/build_base_dataset_from_trace.py \
  --trace ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/nf-trace-rerun.txt \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/task_metrics_base_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/merge_task_metrics.py \
  --base ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/task_metrics_base_rerun.csv \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/per_task_after_audit.csv \
  --c-bytes ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/per_task_c_rerun.csv \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/eager/task_metrics_rerun.csv
```

Use [task_metrics_recorded.csv](eager/task_metrics_recorded.csv) as the main comparison CSV for the preserved run. The additional [recorded_inputs.txt](eager/recorded_inputs.txt) file records the accession split between the run sheet and the broader download log.

The replay-based selective `strace` filter used by `eager`, `rnaseq`, `mag`, `mag_karlsson`, and `chipseq` is [../audit/strace/filter_after_by_task_plan.py](../audit/strace/filter_after_by_task_plan.py).

## RNAseq

The [rnaseq](rnaseq) folder holds the preserved nf-core/rnaseq run. The preserved compact run in this repo produces `78` task rows across `29` task types. The raw `nf-trace` snapshot has `81` rows because it also preserves extra non-compact trace rows.


Command order for `rnaseq`:
- baseline rerun: the `nextflow run` command
- filter the rerun task table by the Experiment 1 `task_plan.csv`, then run the `strace` replay and merge commands that follow

```bash
git clone https://github.com/nf-core/rnaseq.git
cd rnaseq
git checkout 891468c53574d531ae3f75b3a558552839cf973d

nextflow run . -profile test,apptainer \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/configs/slurm_repro.config \
  --camp_output_dir ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq \
  --outdir <results_dir> \
  -work-dir <work_dir> \
  --igenomes_ignore \
  --igenomes_base <dummy_igenomes_dir>

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/filter_after_by_task_plan.py \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/per_task_after_rerun.csv \
  --task-plan <task_plan.csv> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/per_task_after_audit.csv

bash ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/replay_c_bytes_from_after_csv.sh \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/per_task_after_audit.csv \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/per_task_c_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/build_base_dataset_from_trace.py \
  --trace ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/nf-trace-rerun.txt \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/task_metrics_base_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/merge_task_metrics.py \
  --base ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/task_metrics_base_rerun.csv \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/per_task_after_audit.csv \
  --c-bytes ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/per_task_c_rerun.csv \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/rnaseq/task_metrics_rerun.csv
```

Compare the rerun against [task_metrics_recorded.csv](rnaseq/task_metrics_recorded.csv), which is the compact recorded comparison CSV for this preserved run.

## MAG

The [mag](mag) folder holds the preserved nf-core/mag test-profile run. The preserved compact run in this repo produces `25` task rows across `14` task types. The raw `nf-trace` snapshot has `27` rows.

The preserved run metadata records nf-core/mag `v5.4.2` from `https://github.com/nf-core/mag.git`, and the bundled [recorded_inputs.txt](mag/recorded_inputs.txt) file captures the exact upstream test-dataset samplesheet URL plus the `CAT`, `GTDB-Tk`, and `BUSCO` test database URLs used by the run.

Command order for `mag`:
- baseline rerun: the `nextflow run` command
- filter the rerun task table by the Experiment 1 `task_plan.csv`, then run the `strace` replay and merge commands that follow

```bash
git clone https://github.com/nf-core/mag.git
cd mag
git checkout 5dabb0159ac0104885e09f301db22126e8fcb394

nextflow run . -profile test,apptainer \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/configs/slurm_repro.config \
  --camp_output_dir ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag \
  --outdir <results_dir> \
  -work-dir <work_dir> \
  --igenomes_ignore \
  --igenomes_base <dummy_igenomes_dir>

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/filter_after_by_task_plan.py \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/per_task_after_rerun.csv \
  --task-plan <task_plan.csv> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/per_task_after_audit.csv

bash ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/replay_c_bytes_from_after_csv.sh \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/per_task_after_audit.csv \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/per_task_c_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/build_base_dataset_from_trace.py \
  --trace ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/nf-trace-rerun.txt \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/task_metrics_base_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/merge_task_metrics.py \
  --base ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/task_metrics_base_rerun.csv \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/per_task_after_audit.csv \
  --c-bytes ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/per_task_c_rerun.csv \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag/task_metrics_rerun.csv
```

## MAG Karlsson

The [mag_karlsson](mag_karlsson) folder holds the preserved Karlsson-derived nf-core/mag run. In this repo snapshot, this run appears only in the held-out Table IV-style evaluation artifacts, where the preserved compact output contains `11` task rows across `5` task types. The raw `nf-trace` snapshot has `15` rows.


Command order for `mag_karlsson`:
- baseline rerun: download local FASTQs, rewrite the sample sheet, then run the workflow
- filter the rerun task table by the Experiment 1 `task_plan.csv`, then run the `strace` replay and merge commands that follow

```bash
git clone https://github.com/nf-core/mag.git
cd mag
git checkout 5dabb0159ac0104885e09f301db22126e8fcb394

# download the three FASTQ pairs used by samplesheet_remote.csv and place them in <karlsson_fastq_dir>
python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/scripts/rewrite_samplesheet_paths.py \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/samplesheet_remote.csv \
  --data-dir <karlsson_fastq_dir> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/samplesheet_local.csv

nextflow run . -profile apptainer \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/configs/slurm_repro.config \
  --camp_output_dir ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson \
  --outdir <results_dir> \
  -work-dir <work_dir> \
  --igenomes_ignore \
  --igenomes_base <dummy_igenomes_dir> \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/samplesheet_local.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/filter_after_by_task_plan.py \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/per_task_after_rerun.csv \
  --task-plan <task_plan.csv> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/per_task_after_audit.csv

bash ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/replay_c_bytes_from_after_csv.sh \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/per_task_after_audit.csv \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/per_task_c_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/build_base_dataset_from_trace.py \
  --trace ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/nf-trace-rerun.txt \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/task_metrics_base_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/merge_task_metrics.py \
  --base ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/task_metrics_base_rerun.csv \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/per_task_after_audit.csv \
  --c-bytes ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/per_task_c_rerun.csv \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mag_karlsson/task_metrics_rerun.csv
```

## Chipseq

The [chipseq](chipseq) folder holds the preserved nf-core/chipseq run. The preserved compact run in this repo contains `215` task rows across `50` task types.

The preserved run metadata records nf-core/chipseq `v2.1.0` at commit `76e2382b6d443db4dc2396e6831d1243256d80b0`. The bundled [samplesheet_valid_remote.csv](chipseq/samplesheet_valid_remote.csv) is the preserved validated samplesheet from the run.

Command order for `chipseq`:
- baseline rerun: the `nextflow run` command
- filter the rerun task table by the Experiment 1 `task_plan.csv`, then run the `strace` replay and merge commands that follow

```bash
git clone https://github.com/nf-core/chipseq.git
cd chipseq
git checkout 76e2382b6d443db4dc2396e6831d1243256d80b0

nextflow run . -profile test,apptainer \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/configs/slurm_repro.config \
  --camp_output_dir ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq \
  --outdir <results_dir> \
  -work-dir <work_dir> \
  --igenomes_ignore \
  --igenomes_base <dummy_igenomes_dir>

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/filter_after_by_task_plan.py \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/per_task_after_rerun.csv \
  --task-plan <task_plan.csv> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/per_task_after_audit.csv

bash ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/replay_c_bytes_from_after_csv.sh \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/per_task_after_audit.csv \
  ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/per_task_c_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/build_base_dataset_from_trace.py \
  --trace ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/nf-trace-rerun.txt \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/task_metrics_base_rerun.csv

python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/strace/merge_task_metrics.py \
  --base ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/task_metrics_base_rerun.csv \
  --after ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/per_task_after_audit.csv \
  --c-bytes ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/per_task_c_rerun.csv \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/chipseq/task_metrics_rerun.csv
```

## PyRadiomics

The [pyradiomics](pyradiomics) folder holds the preserved strict PyRadiomics / TotalSegmentator run. For this workflow, the runnable workflow source needed for reproduction is now bundled directly in the artifact folder itself: [Snakefile](pyradiomics/Snakefile), [scripts/profile_task.py](pyradiomics/scripts/profile_task.py), [scripts/validate_case.py](pyradiomics/scripts/validate_case.py), [scripts/extract_feature_class.py](pyradiomics/scripts/extract_feature_class.py), [scripts/merge_feature_classes.py](pyradiomics/scripts/merge_feature_classes.py), [scripts/merge_all_samples.py](pyradiomics/scripts/merge_all_samples.py), and [scripts/aggregate_task_metrics.py](pyradiomics/scripts/aggregate_task_metrics.py). No separate GitHub clone step is required for this PyRadiomics artifact section.

To make the exact historical strict run reproducible from the current snapshot, [usable_samples_32k_remote.tsv](pyradiomics/usable_samples_32k_remote.tsv) was reconstructed from the `3556` preserved validation outputs under `results/validated`. The copied [PYRADIOMICS_RUN_END_TO_END.md](pyradiomics/PYRADIOMICS_RUN_END_TO_END.md) file keeps the original `/shared/...` cluster paths from the preserved run history, so please treat that document as historical provenance and use the artifact-local commands below for reproduction.

Dataset download info: download the TotalSegmentator v201 dataset from `https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip` and extract it so the case directories appear directly under `<totalseg_data_dir>`. The strict run expects CT volumes plus five ROI masks per case: `liver`, `spleen`, `kidney_left`, `kidney_right`, and `pancreas`. In the local snapshot used to assemble this artifact, the dataset root contained `1215` case directories. The preserved strict-invalid rows are listed in [strict_invalid_samples.tsv](pyradiomics/strict_invalid_samples.tsv).

Provide `Snakemake`, `Python 3`, `PyRadiomics`, `SimpleITK`, `strace`, and a `Slurm` environment before running the commands below. The output files created by the bundled workflow are `results/all_features.csv`, `results/task_metrics_detailed.csv`, `results/task_metrics_sizey_like.csv`, and `results/system_config.json`.

```bash
# 1. download and extract the dataset
wget -O Totalsegmentator_dataset_v201.zip \
  https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip
unzip Totalsegmentator_dataset_v201.zip -d <totalseg_data_dir>

# 2. move to the PyRadiomics artifact folder
cd IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/pyradiomics

# 3. rewrite the reconstructed exact run sheet so it points to your local TotalSegmentator dataset root
python scripts/rewrite_samplesheet_paths.py \
  --input usable_samples_32k_remote.tsv \
  --data-dir <totalseg_data_dir> \
  --output usable_samples_32k_local.tsv

# 4. run the preserved strict workflow over the reconstructed exact 3556-row run sheet
snakemake \
  --snakefile Snakefile \
  --configfile config_32k_repro.yaml \
  --profile profiles/slurm \
  -j 640
```

The command sequence above is complete in this artifact snapshot once the dataset is available under `<totalseg_data_dir>`. After the rerun, please compare the regenerated output counts against [recorded_output_summary.csv](pyradiomics/recorded_output_summary.csv). The historical top-level result counts are documented in [PYRADIOMICS_RUN_END_TO_END.md](pyradiomics/PYRADIOMICS_RUN_END_TO_END.md), while the artifact-local run should regenerate the corresponding files under `workflows/pyradiomics/results/`.

## Bowtie

The [bowtie](bowtie) folder holds the preserved Experiment 2 `bowtie2_audit_nf` workflow. For this workflow, the runnable source needed for reproduction is bundled directly in the artifact folder itself: [main.nf](bowtie/main.nf), [nextflow_repro.config](bowtie/nextflow_repro.config), [data/public_runs_prjeb11501_20.tsv](bowtie/data/public_runs_prjeb11501_20.tsv), [scripts/split_paired_fastq.py](bowtie/scripts/split_paired_fastq.py), and [bin/merge_ebpf_trace.py](bowtie/bin/merge_ebpf_trace.py).

Dataset download info: the workflow uses the public `PRJEB11501` yeast paired-end manifest in [data/public_runs_prjeb11501_20.tsv](bowtie/data/public_runs_prjeb11501_20.tsv). The reference FASTA is `https://ftp.ensembl.org/pub/current_fasta/saccharomyces_cerevisiae/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz`.

Provide `Nextflow`, `Java`, `Python 3`, `wget`, `gzip`, `bowtie2`, `samtools`, and a Linux `Slurm` environment before running the commands below. The baseline run does not need root. The selective eBPF rerun does require root execution permission and BCC support on the execution nodes, and it expects a gating task plan generated from the current baseline prediction output. The workflow itself generates the trace-level outputs under `<results_dir>/pipeline_info`; the compact paper-level CSV included in this artifact is [task_metrics_recorded.csv](bowtie/task_metrics_recorded.csv), which was extracted from the preserved Experiment 2 input table in [../results/predictions_qlgbm_exp2_all.csv](../results/predictions_qlgbm_exp2_all.csv).

```bash
# 1. move to the Bowtie artifact folder
cd IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/bowtie

# 2. baseline workflow rerun without eBPF
nextflow run . -profile slurm \
  -c nextflow_repro.config \
  --outdir <baseline_results_dir> \
  --metrics_dir <baseline_metrics_dir> \
  --work_dir <baseline_work_dir>

# 3. selective eBPF rerun for the task instances marked Audit in <task_plan.csv>
nextflow run . -profile slurm \
  -c nextflow_repro.config \
  -c conf/ebpf_selective_overlay.config \
  --camp_task_plan <task_plan.csv> \
  --camp_run_with_optional_audit ../../audit/ebpf/run_with_optional_audit.py \
  --camp_ebpf_tracer ../../audit/ebpf/ebpf_audit.py \
  --outdir <results_dir> \
  --metrics_dir <metrics_dir> \
  --work_dir <work_dir>
```

Command order for `bowtie`:
- baseline rerun: verify the workflow runs and writes the expected alignment outputs
- after the gating logic marks task instances as `Audit`, use the eBPF rerun to regenerate `pipeline_info/execution_trace.txt`, `pipeline_info/merged_audit_trace.csv`, and `pipeline_info/merged_audit_trace.tsv` under `<results_dir>`

Compare the regenerated raw merged table against [merged_audit_trace_recorded.csv](bowtie/merged_audit_trace_recorded.csv) and [raw_task_metrics_summary.csv](bowtie/raw_task_metrics_summary.csv). The preserved compact paper-level table [task_metrics_recorded.csv](bowtie/task_metrics_recorded.csv) remains the paper-side comparison CSV with `2928` rows across `7` task types. The retry-expanded raw preserved run history is larger, at `3058` rows.

## MCMICRO

The [mcmicro](mcmicro) folder holds the preserved Experiment 2 `mcmicro` workflow. For this workflow, the artifact preserves the exact historical public sample sheet layout, the exact marker sheet, the compact per-task table used by Experiment 2, and the public image provenance reflected by the preserved run metadata.

Dataset download info: the preserved run evidence points to two public CyCIF OME-TIFF images:
- `https://raw.githubusercontent.com/nf-core/test-datasets/modules/data/imaging/ome-tiff/cycif-tonsil-cycle1.ome.tif`
- `https://raw.githubusercontent.com/nf-core/test-datasets/modules/data/imaging/ome-tiff/cycif-tonsil-cycle2.ome.tif`

The preserved local mirrored run expanded those two public images into `420` public sample IDs, listed in [public_sample_ids.csv](mcmicro/public_sample_ids.csv), from `PUBLIC_0001` through `PUBLIC_0420`. The exact preserved mirrored layout is bundled as [samplesheet_public_cycif_5k_remote.csv](mcmicro/samplesheet_public_cycif_5k_remote.csv), which contains `840` rows and maps each `PUBLIC_XXXX` sample to the two historical `/shared/mcmicro/public_data/cycif_5k/` image paths. The preserved marker sheet used by the run is [markers_test_full.csv](mcmicro/markers_test_full.csv).

The preserved run metadata and local source snapshot identify this workflow as `nf-core/mcmicro`. The bundled [recorded_software_versions.yml](mcmicro/recorded_software_versions.yml) records `Workflow: nf-core/mcmicro: v2.0.0`. The exact public commit hash is not pinned in this artifact folder, so the command sequence below preserves the recorded inputs and parameters while cloning the upstream `nf-core/mcmicro` source.

The baseline `mcmicro` rerun does not need root. The selective eBPF rerun does require root execution permission plus BCC support on the Linux compute nodes, and it uses the shared wrapper and tracer documented in [../audit/ebpf/README.md](../audit/ebpf/README.md). It also expects a gating task plan generated from the current baseline prediction output.

Sequence for `mcmicro` from the current artifact snapshot:

```bash
# 1. clone the upstream workflow source used by the preserved run family
git clone https://github.com/nf-core/mcmicro.git
cd mcmicro

# 2. download the two preserved public CyCIF images
mkdir -p <mcmicro_data_dir>
wget -O <mcmicro_data_dir>/cycif-tonsil-cycle1.ome.tif \
  https://raw.githubusercontent.com/nf-core/test-datasets/modules/data/imaging/ome-tiff/cycif-tonsil-cycle1.ome.tif
wget -O <mcmicro_data_dir>/cycif-tonsil-cycle2.ome.tif \
  https://raw.githubusercontent.com/nf-core/test-datasets/modules/data/imaging/ome-tiff/cycif-tonsil-cycle2.ome.tif

# 3. rewrite the preserved historical sample sheet so it points to your local image directory
python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/scripts/rewrite_samplesheet_paths.py \
  --input ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/samplesheet_public_cycif_5k_remote.csv \
  --data-dir <mcmicro_data_dir> \
  --output ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/samplesheet_public_cycif_5k_local.csv

# 4. baseline workflow rerun without eBPF
nextflow run . -profile apptainer \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/configs/slurm_5nodes_exclusive.config \
  --input_cycle ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/samplesheet_public_cycif_5k_local.csv \
  --marker_sheet ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/markers_test_full.csv \
  --illumination basicpy \
  --segmentation mesmer,cellpose \
  --backsub \
  --outdir <baseline_results_dir> \
  -work-dir <baseline_work_dir>

# 5. selective eBPF rerun for the task instances marked Audit in <task_plan.csv>
nextflow run . -profile apptainer \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/configs/slurm_5nodes_exclusive.config \
  -c ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/configs/ebpf_apptainer_selective_artifact.config \
  --camp_task_plan <task_plan.csv> \
  --camp_run_with_optional_audit ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/ebpf/run_with_optional_audit.py \
  --camp_ebpf_tracer ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/ebpf/ebpf_audit.py \
  --input_cycle ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/samplesheet_public_cycif_5k_local.csv \
  --marker_sheet ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/markers_test_full.csv \
  --illumination basicpy \
  --segmentation mesmer,cellpose \
  --backsub \
  --outdir <results_dir> \
  -work-dir <work_dir>

# 6. merge the raw eBPF task table
python ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/audit/ebpf/merge_mcmicro_ebpf_task_metrics.py \
  --outdir <results_dir> \
  --work-dir <work_dir> \
  --csv-out ../IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/mcmicro/task_metrics_with_ebpf_rerun.csv
```

Command order for `mcmicro`:
- baseline rerun: verify the workflow runs correctly over the preserved `420`-sample public CyCIF layout
- after the gating logic marks task instances as `Audit`, use the eBPF rerun to regenerate `task_metrics_with_ebpf_rerun.csv` and compare it against [task_metrics_with_ebpf_recorded.csv](mcmicro/task_metrics_with_ebpf_recorded.csv)

Treat the preserved compact table [task_metrics_recorded.csv](mcmicro/task_metrics_recorded.csv) as the exact Experiment 2 `mcmicro` row set used by the CAMP artifact. That compact table was extracted from [../results/predictions_qlgbm_exp2_all.csv](../results/predictions_qlgbm_exp2_all.csv). The compact table contains `8330` rows across `9` task types, while the preserved retry-expanded raw eBPF table contains `8385` rows. The largest raw-versus-compact difference is in `DEEPCELL_MESMER`, as summarized in [raw_task_metrics_summary.csv](mcmicro/raw_task_metrics_summary.csv) and [task_metrics_summary.csv](mcmicro/task_metrics_summary.csv).

## Minimap

The [minimap](minimap) folder holds the preserved `minimap2_audit_nf` workflow. For this workflow, the runnable source needed for reproduction is bundled directly in the artifact folder itself: [main.nf](minimap/main.nf), [nextflow_repro.config](minimap/nextflow_repro.config), [scripts/generate_reference.py](minimap/scripts/generate_reference.py), [scripts/plan_windows.py](minimap/scripts/plan_windows.py), [scripts/generate_window_reads.py](minimap/scripts/generate_window_reads.py), [conf/ebpf_overlay.config](minimap/conf/ebpf_overlay.config), the shared tracer in [../audit/ebpf](../audit/ebpf), and [bin/merge_ebpf_trace.py](minimap/bin/merge_ebpf_trace.py).

Dataset download info: this workflow does not download an external dataset. It generates a deterministic synthetic long-read benchmark from the bundled scripts. The reference generator creates four chromosomes, `chr1`, `chr2`, `chr3`, and `chrX`, with segmental duplications and tandem-repeat expansion annotations. The manifest generator then creates `1250` windows with a deterministic mix of `unique_window`, `segdup_window`, and `tr_window` tasks.

Provide `Nextflow`, `Java`, `Python 3`, `minimap2`, `samtools`, and a Linux `Slurm` environment before running the commands below. The baseline run does not need root. The selective eBPF rerun does require root execution permission and BCC support on the execution nodes, and it expects a gating task plan generated from the current baseline prediction output.

```bash
# 1. move to the Minimap artifact folder
cd IEEE_CLUSTER_2026_CAMP_ARTIFACT/workflows/minimap

# 2. baseline workflow rerun without eBPF
nextflow run . -profile slurm \
  -c nextflow_repro.config \
  --outdir <baseline_results_dir> \
  --metrics_dir <baseline_metrics_dir> \
  --work_dir <baseline_work_dir>

# 3. selective eBPF rerun for the task instances marked Audit in <task_plan.csv>
nextflow run . -profile slurm \
  -c nextflow_repro.config \
  -c conf/ebpf_selective_overlay.config \
  --camp_task_plan <task_plan.csv> \
  --camp_run_with_optional_audit ../../audit/ebpf/run_with_optional_audit.py \
  --camp_ebpf_tracer ../../audit/ebpf/ebpf_audit.py \
  --outdir <results_dir> \
  --metrics_dir <metrics_dir> \
  --work_dir <work_dir>
```

Command order for `minimap`:
- baseline rerun: verify the synthetic reference, manifests, reads, BAMs, and flagstat outputs are generated
- after the gating logic marks task instances as `Audit`, use the eBPF rerun to regenerate `pipeline_info/execution_trace.txt`, `pipeline_info/merged_audit_trace.csv`, and `pipeline_info/merged_audit_trace.tsv`

Compare the regenerated raw merged table against [merged_audit_trace_recorded.csv](minimap/merged_audit_trace_recorded.csv) and [raw_task_metrics_summary.csv](minimap/raw_task_metrics_summary.csv). The preserved compact paper-level table [task_metrics_recorded.csv](minimap/task_metrics_recorded.csv), extracted from [../results/predictions_qlgbm_exp2_all.csv](../results/predictions_qlgbm_exp2_all.csv), contains `2385` rows across `6` task types. The preserved retry-expanded raw table contains `5032` rows.
