nextflow.enable.dsl=2

process GENERATE_REFERENCE {
    tag 'genome'
    cpus params.ref_cpus
    memory params.ref_memory
    time params.ref_time
    publishDir "${params.outdir}/reference", mode: 'rellink'

    output:
    path 'genome.fa', emit: ref_fa
    path 'genome_annotations.bed', emit: ref_bed

    script:
    """
    python3 ${workflow.projectDir}/scripts/generate_reference.py \
      --out-fa genome.fa \
      --out-bed genome_annotations.bed
    """
}

process PLAN_WINDOWS {
    tag "n${n_windows}"
    cpus params.plan_cpus
    memory params.plan_memory
    time params.plan_time
    publishDir "${params.outdir}/manifests", mode: 'rellink'

    input:
    val n_windows

    output:
    path 'window_manifest.tsv', emit: manifest

    script:
    """
    python3 ${workflow.projectDir}/scripts/plan_windows.py \
      --n-windows ${n_windows} \
      --out window_manifest.tsv
    """
}

process BUILD_INDEX {
    tag 'genome_mmi'
    cpus params.index_cpus
    memory params.index_memory
    time params.index_time
    publishDir "${params.outdir}/reference", mode: 'rellink'

    input:
    path ref_fa

    output:
    path 'genome.mmi', emit: mmi
    path 'genome.fa.fai', emit: fai

    script:
    """
    minimap2 -x map-ont -t ${task.cpus} -d genome.mmi ${ref_fa}
    samtools faidx ${ref_fa}
    """
}

process GENERATE_WINDOW_READS {
    tag "window_${window_id}"
    cpus params.reads_cpus
    memory params.reads_memory
    time params.reads_time
    publishDir "${params.outdir}/reads", mode: 'rellink'

    input:
    tuple val(window_id), val(filename), val(window_type), val(expected_peak_gb), val(seed)
    path ref_fa
    path ref_bed

    output:
    tuple val(window_id), val(filename), val(window_type), val(expected_peak_gb), path(filename), emit: reads

    script:
    """
    python3 ${workflow.projectDir}/scripts/generate_window_reads.py \
      --genome ${ref_fa} \
      --annotations ${ref_bed} \
      --window-id ${window_id} \
      --window-type ${window_type} \
      --seed ${seed} \
      --out ${filename}
    """
}

process MAP_WINDOW {
    tag "window_${window_id}"
    cpus params.map_cpus
    memory params.map_memory
    time params.map_time
    publishDir "${params.outdir}/bam", mode: 'rellink', pattern: '*.bam'

    input:
    tuple val(window_id), val(filename), val(window_type), val(expected_peak_gb), path(reads)
    path mmi

    output:
    tuple val(window_id), val(filename), val(window_type), val(expected_peak_gb), path("window_${window_id.toString().padLeft(4,'0')}.bam"), emit: bam

    script:
    def bam = "window_${window_id.toString().padLeft(4,'0')}.bam"
    """
    minimap2 -a -x map-ont --MD -t ${task.cpus} ${mmi} ${reads} 2> window_${window_id}.minimap2.log \
      | samtools sort -@ ${task.cpus} -o ${bam} -T window_${window_id}.tmp
    """
}

process INDEX_BAM {
    tag "window_${window_id}"
    cpus params.bam_index_cpus
    memory params.bam_index_memory
    time params.bam_index_time
    publishDir "${params.outdir}/bam", mode: 'rellink', pattern: '*.bai'

    input:
    tuple val(window_id), val(filename), val(window_type), val(expected_peak_gb), path(bam)

    output:
    tuple val(window_id), val(filename), val(window_type), val(expected_peak_gb), path(bam), path("${bam}.bai"), emit: indexed_bam

    script:
    """
    samtools index ${bam}
    """
}

process FLAGSTAT_BAM {
    tag "window_${window_id}"
    cpus params.flagstat_cpus
    memory params.flagstat_memory
    time params.flagstat_time
    publishDir "${params.outdir}/flagstat", mode: 'rellink'

    input:
    tuple val(window_id), val(filename), val(window_type), val(expected_peak_gb), path(bam), path(bai)

    output:
    path "window_${window_id.toString().padLeft(4,'0')}.flagstat.txt", emit: flagstat

    script:
    def out = "window_${window_id.toString().padLeft(4,'0')}.flagstat.txt"
    """
    samtools flagstat ${bam} > ${out}
    """
}

workflow {
    ref = GENERATE_REFERENCE()
    manifest = PLAN_WINDOWS(params.n_windows)
    index = BUILD_INDEX(ref.ref_fa)

    window_meta = manifest.manifest
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            tuple(
                row.window_id as Integer,
                row.filename.toString(),
                row.window_type.toString(),
                row.expected_peak_gb.toString(),
                row.seed as Long
            )
        }

    reads = GENERATE_WINDOW_READS(window_meta, ref.ref_fa, ref.ref_bed)
    mapped = MAP_WINDOW(reads.reads, index.mmi)
    indexed = INDEX_BAM(mapped.bam)
    FLAGSTAT_BAM(indexed.indexed_bam)
}

workflow.onComplete {
    def mergeCmd = [
        'python3',
        "${workflow.projectDir}/bin/merge_ebpf_trace.py",
        '--outdir', params.outdir.toString(),
        '--work-dir', params.work_dir.toString(),
        '--trace', params.trace_file.toString(),
        '--csv-out', params.merged_csv.toString(),
        '--tsv-out', params.merged_tsv.toString()
    ].collect { it.toString() }

    println "[onComplete] merging audit + trace metrics"
    def pb = new ProcessBuilder(mergeCmd)
    pb.redirectErrorStream(true)
    def proc = pb.start()
    proc.inputStream.eachLine { println it }
    def rc = proc.waitFor()
    println "[onComplete] merge exit code=${rc}"
}
