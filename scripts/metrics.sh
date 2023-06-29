
#!/bin/bash
source scripts/config.sh
source scripts/env.sh # get the following variables

cd tex
metrics="metrics"


echo "
-------------------------
START generating $metrics PDFs for: 
$PROJECT_ID,
-------------------------"

mkdir -p $metrics

END=2
for k in $(seq 1 $END); do 
    pdflatex --interaction=nonstopmode "\newcommand\plots{$VIS_DIR}\newcommand\block{$PROJECT_ID}\input{$metrics}"
done
mv ${metrics}.pdf $metrics/${metrics}.pdf

rm ${metrics}.out ${metrics}.log ${metrics}.aux

echo "DONE!"
