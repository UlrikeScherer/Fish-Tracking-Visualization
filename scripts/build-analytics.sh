
#!/bin/bash
source scripts/config.sh
if [ ! -d "$VIS_DIR" ]; then
    echo "ERROR $VIS_DIR does not exists, first generate the plots with the python script"
    exit 1
fi
source scripts/env.sh # get the following variables

cd tex
position=("front_1" "front_2" "back_1" "back_2")
analytics="analytics"


echo "
-------------------------
START generating analytics PDFs for: 
$BLOCK
-------------------------"
TIME_INTERVAL=100
if [ $1 ]; then 
    TIME_INTERVAL=$1
fi 

mkdir -p $analytics/$BLOCK


for b in ${!position[@]}; do
    echo -e "pdf for ${position[$b]} \n"

    END=2
    for k in $(seq 1 $END); do 
        pdflatex --interaction=nonstopmode "\newcommand\plots{$VIS_DIR}\newcommand\block{$BLOCK}\newcommand\position{${position[$b]}}\newcommand\timeinterval{$TIME_INTERVAL}\input{analytics}"

    done
    mv ${analytics}.pdf $analytics/$BLOCK/${TIME_INTERVAL}_${analytics}_${position[$b]}.pdf
done 

rm ${analytics}.out ${analytics}.log ${analytics}.aux

echo "DONE!"