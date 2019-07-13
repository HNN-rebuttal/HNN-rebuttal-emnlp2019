#!/bin/bash
echo 'procrustes+NN  hinge+NN procrustes+HNN' > .tmp
paste <(grep "P@1(%)" l2.out | awk -F, '{print $1}' | awk '{print $NF}') <(grep "P@1(%)" hinge.out | awk -F, '{print $1}' | awk '{print $NF}') <(grep "P@1(%)" l2_hnn.out | awk -F, '{print $1}' | awk '{print $NF}') >> .tmp 
gnuplot -p -e "set xlabel 'frequency bin'; set ylabel 'accuracy (%)'; plot for [col=1:3] '.tmp' using 0:col with lines title columnheader"
rm .tmp
