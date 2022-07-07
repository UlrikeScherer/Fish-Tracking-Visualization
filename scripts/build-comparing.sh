
#!/bin/bash
source scripts/config.sh
source scripts/env.sh # get the following variables

cd tex
comparing="comparing_results_of_tracesizes"
clustering="clustering"


echo "
-------------------------
START generating clustering trace size comparing PDFs for: 
$BLOCK
-------------------------"

mkdir -p $clustering/$BLOCK

END=2
for k in $(seq 1 $END); do 
    pdflatex --interaction=nonstopmode "\newcommand\plots{$VIS_DIR}\newcommand\block{$BLOCK}\input{$comparing}"
    #pdflatex "\newcommand\block{$BLOCK}\newcommand\tracesize{$TRACE_SIZE}\input{$clustering}"
done
mv ${comparing}.pdf $clustering/$BLOCK/${BLOCK}_${comparing}.pdf

rm ${comparing}.out ${comparing}.log ${comparing}.aux

echo "DONE!"