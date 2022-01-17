#!/bin/bash

TEST=$1
cameras=('23520289' '23484201' '23520258' '23442333' '23520268' '23520257' '23520266' '23484204' '23520278' '23520276' '23520270' '23520264')
position=("front" "back")
PREFIX="file://" #run:

source env.sh

echo "$BLOCK, $rootserver, $path_recordings, $path_csv"

mkdir trajectory

if [ "$TEST" == "test" ]; then
    cameras=('23520289') 
    position=("back")
fi


for b in ${!position[@]}; do
    echo -e "pdf for ${position[$b]} \n"
    for i in ${!cameras[@]}; do
        secff="$(ls -d $path_csv/FE_${BLOCK}_autotracks_${position[$b]}/${cameras[$i]}/*.${cameras[$i]}/ | sort -V | head -1 | sed 's/.*1550\([^.]*\).*/\1/')"
        foldersmp4="$(ls -d $path_recordings/FE_${BLOCK}_recordings_*/${cameras[$i]}/*.${cameras[$i]}*/ | sort -V)"
        filesmp4="$(ls $path_recordings/FE_${BLOCK}_recordings_*/${cameras[$i]}/*.${cameras[$i]}*/*.mp4 | sort -V)"

        texheader="%\usepackage{etoolbox}
                    %% root folders: ---------------------
                    \newcommand\rootserver{$rootserver}
                    \newcommand\rootcsv{$path_csv}
                    \newcommand\rootrecord{$path_recordings}
                    \newcommand\block{$BLOCK}
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
                    \newcommand\addsub[1]{%
                        \csdef{sub\thecntsub}{#1}%
                        \stepcounter{cntsub}}
                    \newcommand\getsub[1]{%
                        \csuse{sub#1}}
                    % for csv -----------
                    \newcounter{cntcsv}
                    \newcommand\csvlist{}
                    \newcommand\addcsv[1]{%
                        \csdef{csv\thecntcsv}{#1}%
                        \stepcounter{cntcsv}}
                    \newcommand\getcsv[1]{%
                        \csuse{csv#1}}
                        "

        for f in $foldersmp4; do
            texheader="$texheader \addtext{$PREFIX$f}
            "
        done

        #ffiles=$"(${f}*.mp4 | sort -V)"
        for f in $filesmp4; do 
            texheader="$texheader \addsub{\href{$PREFIX$f}{mp4}}
            "
        done
        
        days="$(ls -d $path_csv/FE_block1_autotracks_${position[$b]}/${cameras[$i]}/*.${cameras[$i]}*/ | sort -V )"
        for d in $days; do 
            d=$(basename $d)
            day=${d: : 24}
            filescsv="$(ls $path_csv/FE_block1_autotracks_${position[$b]}/${cameras[$i]}/*.${cameras[$i]}*/${cameras[$i]}_$day*.csv | sort -V)"
            for f in $filescsv; do 
                texheader="$texheader \addcsv{\href{$PREFIX$f}{csv}}
                "
            done
        done

        echo "$texheader" > arrayoflinks.tex

        END=2
        for k in $(seq 1 $END); do 
            #pdflatex "\newcommand\secfirstplot{$secff}\newcommand\position{${position[$b]}}\newcommand\camera{${cameras[$i]}}\input{main}"
            pdflatex --interaction=nonstopmode "\newcommand\secfirstplot{$secff}\newcommand\position{${position[$b]}}\newcommand\camera{${cameras[$i]}}\input{main}"
        done
        mv main.pdf trajectory/${cameras[$i]}_${position[$b]}.pdf
    done
done
