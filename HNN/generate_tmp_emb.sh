#!/bin/bash
emb_file=$1
vocab_file=$2
out_file=$3
d=$(head -n 1 $emb_file | awk '{print $2}')
n=$(wc -l $vocab_file | awk '{print $1}')
echo $n $d > $out_file
awk -v o=$out_file 'NR==FNR{a[$0];next} FNR>1 && $1 in a {print >> o}' $vocab_file $emb_file
