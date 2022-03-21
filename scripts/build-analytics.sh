
#!/bin/bash
source scripts/env.sh # get the following variables

cd tex
position=("front_1" "front_2" "back_1" "back_2")
analytics="analytics"


echo "
-------------------------
START generating analytics PDFs for: 
$BLOCK
-------------------------"

mkdir $analytics
mkdir $analytics/$BLOCK


for b in ${!position[@]}; do
    echo -e "pdf for ${position[$b]} \n"

    END=2
    for k in $(seq 1 $END); do 
        pdflatex --interaction=nonstopmode "\newcommand\block{$BLOCK}\newcommand\position{${position[$b]}}\input{analytics}"

    done
    mv analytics.pdf $analytics/$BLOCK/${analytics}_${position[$b]}.pdf
done 
