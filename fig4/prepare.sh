#!/bin/bash
# Extract Teff data from public dataset released by Bergot et al. in tsv format
set -ex

SRRS=(
  'SRR1188136' 'SRR1188139' 'SRR1188142'
  'SRR1188146' 'SRR1188167' 'SRR1188171'
)
NCBI_DIR=~/ncbi

for srr in ${SRRS[*]}; do
  if [ -f "input/$srr/clones.txt" ]
  then
    echo "input/$srr/clones.txt found"
  else
    orig_path=$(pwd)
    echo "Generate input/$srr/clones.txt"
    
    prefetch $srr
    dst_dir=input/$srr
    mkdir -p $dst_dir
    cd $dst_dir
    
    mv $NCBI_DIR/public/sra/$srr.sra .
    fastq-dump $srr.sra
    mixcr align -f -t 2 --species mmu $srr.fastq $srr.fastq alignments.vdjca
    mixcr assemble -f alignments.vdjca clones.clns
    mixcr exportClones -f -s -c TRB -t -o clones.clns clones.txt
    
    cd $orig_path
  fi
done