
#!/bin/bash
source scripts/config.sh
source scripts/env.sh # get the following variables

cd tex
clustering="clustering"


echo "
-------------------------
START generating clustering PDFs for: 
$BLOCK
-------------------------"
TRACE_SIZE=200
if [ $1 ]; then 
    TRACE_SIZE=$1
fi 

mkdir -p $clustering/$BLOCK

END=2
for k in $(seq 1 $END); do 
    pdflatex --interaction=nonstopmode "\newcommand\plots{$VIS_DIR}\newcommand\block{$BLOCK}\newcommand\tracesize{$TRACE_SIZE}\input{$clustering}"
    #pdflatex "\newcommand\block{$BLOCK}\newcommand\tracesize{$TRACE_SIZE}\input{$clustering}"
done
mv ${clustering}.pdf $clustering/$BLOCK/${TRACE_SIZE}_${clustering}.pdf

rm ${clustering}.out ${clustering}.log ${clustering}.aux

echo "DONE!"