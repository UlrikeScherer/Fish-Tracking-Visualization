cameras=('23520289' '23484201' '23520258')
position=("front" "back")
rootserver="/Volumes/data/loopbio_data/1_FE_(fingerprint_experiment)_SepDec2021/FE_block1"
for b in ${!position[@]}; do
    echo "pdf for ${position[$b]} \n"
    for i in ${!cameras[@]}; do
        array="$(ls -d $rootserver/FE*$b/${cameras[$i]}/*.${cameras[$i]}/ | sort -V)"

        END=2
        for k in $(seq 1 $END); do 
            pdflatex "\newcommand\secfirstplot{${array[0]: -2}}\newcommand\position{${position[$b]}}\newcommand\camera{${cameras[$i]}}\input{main}"
        done
        mv main.pdf ${cameras[$i]}_${position[$b]}.pdf
    done
done