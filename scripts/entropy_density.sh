
#!/bin/bash
source scripts/config.sh
source scripts/env.sh # get the following variables

cd tex
entropy_density="entropy_density"


echo "
-------------------------
START generating $entropy_density PDFs for: 
$BLOCK
-------------------------"

mkdir -p $entropy_density/$BLOCK

END=2
for k in $(seq 1 $END); do 
    pdflatex --interaction=nonstopmode "\newcommand\plots{$VIS_DIR}\newcommand\block{$BLOCK}\input{$entropy_density}"
done
mv ${entropy_density}.pdf $entropy_density/$BLOCK/${entropy_density}_$BLOCK.pdf

rm ${entropy_density}.out ${entropy_density}.log ${entropy_density}.aux

echo "DONE!"