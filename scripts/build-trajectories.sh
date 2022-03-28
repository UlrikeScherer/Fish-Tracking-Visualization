#!/bin/bash

source scripts/env.sh # get the following variables

cd tex
cameras=('23520289' '23484201' '23520258' '23442333' '23520268' '23520257' '23520266' '23484204' '23520278' '23520276' '23520270' '23520264')
position=("front" "back")
PREFIX="file://" #run:


STARTTIME=$RECORDINGTIME
CSV_DIR=$path_csv
MAX_IDX_OF_DAY=14
SUBFIGURE_WIDTH="0.24\textwidth"
LEGEND="\trajectorylegend"
FILES="files"
mkdir $FILES

while [[ "$#" -gt 0 ]]; do
    case $1 in
        #-b|--block) block="$2"; shift ;;
        -t|--test) test=1 ;;
        -f|--feeding) feeding=1;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
if [ $feeding ]; then
    STARTTIME=$FEEDINGTIME
    CSV_DIR=$path_csv_feeding_local
    MAX_IDX_OF_DAY=7
    SUBFIGURE_WIDTH="0.33\textwidth"
    LEGEND="\feedinglegend"
fi
if [ $test ]; then
    cameras=('23442333') 
    position=("front")
    echo "Testrun using $cameras, position: $position";
fi
echo "
-------------------------
START generating PDFs for: 
$BLOCK,
$rootserver,
$path_recordings,
$CSV_DIR
$STARTTIME
-------------------------"

mkdir -p trajectory/$STARTTIME/$BLOCK

for b in ${!position[@]}; do
    echo -e "pdf for ${position[$b]} \n"
    POSITION_STR="FE_${STARTTIME}_tracks_${BLOCK}_${position[$b]}"
    for i in ${!cameras[@]}; do
        secff="$(ls -d $CSV_DIR/$POSITION_STR/${cameras[$i]}/*.${cameras[$i]}/ | sort -V | head -1 | sed 's/.*1550\([^.]*\).*/\1/')"
        
        texheader="%\usepackage{etoolbox}
                    %% root folders: ---------------------
                    \newcommand\rootserver{$rootserver}
                    \newcommand\rootcsv{$CSV_DIR}
                    \newcommand\rootrecord{$path_recordings}
                    \newcommand\block{$BLOCK}
                    \newcommand\posstr{$POSITION_STR/}
                    \newcommand\starttime{$STARTTIME}
                    \newcommand\maxindex{$MAX_IDX_OF_DAY}
                    \newcommand\subfigwidth{$SUBFIGURE_WIDTH}
                    % ---------------------------------------
                    \newcounter{cnt}
                    \newcommand\textlist{}
                    \newcommand\settext[2]{%
                        \csdef{text#1}{#2}}
                    \newcommand\addtext[1]{%
                        \csdef{text\thecnt}{#1}%
                        \stepcounter{cnt}}
                    \newcommand\gettext[1]{%
                        \csuse{text#1}}
                    % for subplots -------------- 
                    \newcounter{cntsub}
                    \newcommand\sublist{}
                    \newcommand\setsub[2]{%
                        \csdef{sub#1}{#2}}
                    \newcommand\addsub[1]{%
                        \csdef{sub\thecntsub}{#1}%
                        \stepcounter{cntsub}}
                    \newcommand\getsub[1]{%
                        \csuse{sub#1}}
                    % for csv -----------
                    \newcounter{cntcsv}
                    \newcommand\csvlist{}
                    \newcommand\setcsv[2]{%
                        \csdef{csv#1}{#2}}
                    \newcommand\addcsv[1]{%
                        \csdef{csv\thecntcsv}{#1}%
                        \stepcounter{cntcsv}}
                    \newcommand\getcsv[1]{%
                        \csuse{csv#1}}
                        "

        daysarray="$LEGEND"
        days="$(ls -d $CSV_DIR/$POSITION_STR/${cameras[$i]}/*.${cameras[$i]}*/ | sort -V )"

        isFIRST=1;
        for d in $days; do 
            d=$(basename $d)
            day=${d: : 24}
            day_id=${day: : 13}
            if [ "$BLOCK" = "block1" ] && [ "$isFIRST" -eq "1" ] && [ "$feeding" != "1" ]; then
                daysarray="$daysarray \plotdayone{${day: : 13}}
                "
                isFIRST=0;
            else
                daysarray="$daysarray\plotdayupdate{${day: : 13}}
                "
            fi 

            filescsv="$(ls $CSV_DIR/$POSITION_STR/${cameras[$i]}/*.${cameras[$i]}*/${cameras[$i]}_$day*.csv | sort -V)"
            C_i=0
            for f in $filescsv; do 
                texheader="$texheader \setcsv{${day_id}$C_i}{\href{${PREFIX}${f}}{csv}}
                "
                let C_i++
            done

            foldermp4="$(ls -d $path_recordings/${cameras[$i]}/${day}*/ | head )"
            texheader="$texheader \addtext{$PREFIX$foldermp4}
            "

            filesmp4="$(ls $foldermp4/*.mp4 | sort -V)"
            C_i=0
            for f in $filesmp4; do 
                texheader="$texheader \setsub{${day_id}$C_i}{\href{$PREFIX$f}{mp4}}
                "
                let C_i++
            done
        done
        # daysarray=${daysarray%?}
        echo "${daysarray}" > $FILES/days_array.tex
        echo "$texheader" > $FILES/arrayoflinks.tex
        if [ "$feeding" -eq 1 ]; then 
            echo "\input{$FILES/${BLOCK}_feedingtime.tex}" >> $FILES/arrayoflinks.tex
        fi
        END=2
        for k in $(seq 1 $END); do 
            #pdflatex "\newcommand\secfirstplot{$secff}\newcommand\position{${position[$b]}}\newcommand\camera{${cameras[$i]}}\input{main}"
            pdflatex --interaction=nonstopmode "\newcommand\secfirstplot{$secff}\newcommand\position{${position[$b]}}\newcommand\camera{${cameras[$i]}}\input{main}"
        done
        mv main.pdf trajectory/$STARTTIME/$BLOCK/${STARTTIME}_${BLOCK}_${cameras[$i]}_${position[$b]}.pdf
    done
done


echo "Execution time: $SECONDS "