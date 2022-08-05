
#!/bin/bash
source scripts/config.sh
source scripts/env.sh # get the following variables

cd tex
metrics="metrics"


echo "
-------------------------
START generating $metrics PDFs for: 
$BLOCK
-------------------------"

mkdir -p $metrics/$BLOCK

END=2
for k in $(seq 1 $END); do 
    pdflatex --interaction=nonstopmode "\newcommand\plots{$VIS_DIR}\newcommand\block{$BLOCK}\input{$metrics}"
done
mv ${metrics}.pdf $metrics/$BLOCK/${metrics}_$BLOCK.pdf

rm ${metrics}.out ${metrics}.log ${metrics}.aux

echo "DONE!"