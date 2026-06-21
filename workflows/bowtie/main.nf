nextflow.enable.dsl=2

process DOWNLOAD_REFERENCE {
    tag 'sacCer_ref'
    cpus params.ref_cpus
    memory params.ref_memory
    time params.ref_time
    publishDir "${params.outdir}/reference", mode: 'rellink'

    output:
    path 'genome.fa', emit: ref_fa

    script:
    """
    wget -O genome.fa.gz '${params.reference_url}'
    gzip -dc genome.fa.gz > genome.fa
    """
}

process BUILD_INDEX {
    tag 'sacCer_bt2'
    cpus params.index_cpus
    memory params.index_memory
    time params.index_time
    publishDir "${params.outdir}/reference", mode: 'rellink'

    input:
    path ref_fa

    output:
    path 'index', emit: idxdir
    path 'genome.fa.fai', emit: fai

    script:
    """
    mkdir -p index
    bowtie2-build --threads ${task.cpus} ${ref_fa} index/genome
    samtools faidx ${ref_fa}
    """
}

process DOWNLOAD_RUN {
    tag { run_accession }
    cpus params.download_cpus
    memory params.download_memory
    time params.download_time
    publishDir "${params.outdir}/downloads", mode: 'rellink'

    input:
    tuple val(run_accession), val(study_accession), val(fastq1_url), val(fastq2_url), val(fastq1_bytes), val(fastq2_bytes), val(chunks)

    output:
    tuple val(run_accession), val(study_accession), path("${run_accession}_1.fastq.gz"), path("${run_accession}_2.fastq.gz"), val(chunks), emit: run_fastqs

    script:
    """
    wget -O ${run_accession}_1.fastq.gz '${fastq1_url}'
    wget -O ${run_accession}_2.fastq.gz '${fastq2_url}'
    """
}

process SPLIT_RUN {
    tag { run_accession }
    cpus params.split_cpus
    memory params.split_memory
    time params.split_time
    publishDir "${params.outdir}/chunks/${run_accession}", mode: 'rellink'

    input:
    tuple val(run_accession), val(study_accession), path(fq1), path(fq2), val(chunks)

    output:
    path 'chunk_manifest.tsv', emit: chunk_manifest

    script:
    """
    python3 ${workflow.projectDir}/scripts/split_paired_fastq.py \
      --run-accession ${run_accession} \
      --fastq1 ${fq1} \
      --fastq2 ${fq2} \
      --chunks ${chunks} \
      --outdir chunks \
      --manifest chunk_manifest.tsv
    """
}

process ALIGN_CHUNK {
    tag { "${run_accession}_chunk_${chunk_id}" }
    cpus params.align_cpus
    memory params.align_memory
    time params.align_time
    publishDir "${params.outdir}/bam", mode: 'rellink', pattern: '*.bam'

    input:
    tuple val(run_accession), val(chunk_id), path(fq1), path(fq2)
    path idxdir

    output:
    tuple val(run_accession), val(chunk_id), path("${run_accession}.chunk_${chunk_id.toString().padLeft(3,'0')}.bam"), emit: bam

    script:
    def bam = "${run_accession}.chunk_${chunk_id.toString().padLeft(3,'0')}.bam"
    """
    bowtie2 \
      -x ${idxdir}/genome \
      -1 ${fq1} -2 ${fq2} \
      --very-sensitive \
      --no-mixed --no-discordant \
      -p ${task.cpus} \
      2> ${run_accession}.chunk_${chunk_id}.bowtie2.log \
    | samtools sort -@ ${task.cpus} -o ${bam} -T ${run_accession}.chunk_${chunk_id}.tmp
    """
}

process INDEX_BAM {
    tag { "${run_accession}_chunk_${chunk_id}" }
    cpus params.bam_index_cpus
    memory params.bam_index_memory
    time params.bam_index_time
    publishDir "${params.outdir}/bam", mode: 'rellink', pattern: '*.bai'

    input:
    tuple val(run_accession), val(chunk_id), path(bam)

    output:
    tuple val(run_accession), val(chunk_id), path(bam), path("${bam}.bai"), emit: indexed_bam

    script:
    """
    samtools index ${bam}
    """
}

process FLAGSTAT_BAM {
    tag { "${run_accession}_chunk_${chunk_id}" }
    cpus params.flagstat_cpus
    memory params.flagstat_memory
    time params.flagstat_time
    publishDir "${params.outdir}/flagstat", mode: 'rellink'

    input:
    tuple val(run_accession), val(chunk_id), path(bam), path(bai)

    output:
    path "${run_accession}.chunk_${chunk_id.toString().padLeft(3,'0')}.flagstat.txt", emit: flagstat

    script:
    def out = "${run_accession}.chunk_${chunk_id.toString().padLeft(3,'0')}.flagstat.txt"
    """
    samtools flagstat ${bam} > ${out}
    """
}

workflow {
    ref = DOWNLOAD_REFERENCE()
    index = BUILD_INDEX(ref.ref_fa)

    runs = Channel
        .fromPath(params.run_manifest)
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            tuple(
                row.run_accession.toString(),
                row.study_accession.toString(),
                row.fastq1_url.toString(),
                row.fastq2_url.toString(),
                row.fastq1_bytes.toString(),
                row.fastq2_bytes.toString(),
                (row.chunks ?: params.chunks_per_run).toString()
            )
        }

    downloaded = DOWNLOAD_RUN(runs)
    split = SPLIT_RUN(downloaded.run_fastqs)

    chunk_meta = split.chunk_manifest
        .splitCsv(header: true, sep: '\t')
        .map { row ->
            tuple(
                row.run_accession.toString(),
                row.chunk_id as Integer,
                file(row.fastq1.toString()),
                file(row.fastq2.toString())
            )
        }

    aligned = ALIGN_CHUNK(chunk_meta, index.idxdir)
    indexed = INDEX_BAM(aligned.bam)
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
