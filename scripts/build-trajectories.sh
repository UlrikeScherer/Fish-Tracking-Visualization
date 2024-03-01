#!/bin/bash
source fishproviz/config.env # get the following variables
PLOTS_TRAJECTORY=$path_csv_local/$PLOTS_DIR
if [ ! -d "$PLOTS_TRAJECTORY" ]; then
    echo "ERROR $PLOTS_TRAJECTORY does not exists, first generate the plots with the python script"
    exit 1
fi
# continue inside the tex directory
cd tex
position=("front" "back")
POS_STRINGS=($POSITION_STR_FRONT $POSITION_STR_BACK)
PREFIX="file://" #run:

PROGRAMNAME=$P_TRAJECTORY
CSV_DIR=$path_csv
LEGEND="\trajectorylegend"
FILES="$PLOTS_TRAJECTORY/$TEX_DIR"
mkdir $FILES

while [[ "$#" -gt 0 ]]; do
    if [[ "$set_cam" == "1" ]]; then
        cameras=("$1")
        set_cam=2
        echo "cam is $cameras"
    else
        case $1 in
            #-b|--block) block="$2"; shift ;;
            -t|--test) test=1 ;;
            -f|--feeding) feeding=1;;
            -l|--local) local=1;;
            -cam|--cam-id) set_cam=1;;
            *) echo "Unknown parameter passed: $1"; exit 1 ;;
        esac
    fi
    shift

done
if [ $feeding ]; then
    PROGRAMNAME=$P_FEEDING
    SUBFIGURE_WIDTH="0.33\textwidth"
    SUBFIGURE_HEIGHT="0.33\textheight"
    LEGEND="\feedinglegend"
fi
if [ $test ]; then
    cameras=('23442333')
    position=("front")
    echo "Testrun using $cameras, position: $position";
fi
if [ $local ]; then
    CSV_DIR=$path_csv_local
fi

SQRT_N=$(echo "sqrt("$MAX_BATCH_IDX+2-$MIN_BATCH_IDX")" | bc -l)
FIG_WIDTH=$(perl -w -e "use POSIX; print 0.95/ceil($SQRT_N/1.0), qq{\n}")
SUBFIGURE_WIDTH="${FIG_WIDTH}\textwidth"
SUBFIGURE_HEIGHT="${FIG_WIDTH}\textheight"
echo "
-------------------------
START generating PDFs for:
$BLOCK,
$rootserver,
$path_recordings,
$PLOTS_TRAJECTORY,
$CSV_DIR,
$PROGRAMNAME,
$MIN_BATCH_IDX,
$MAX_BATCH_IDX,
$SQRT_N,
$FIG_WIDTH
-------------------------"

directory_of_run=$path_csv_local/$VIS_DIR/$PROGRAMNAME/$BLOCK
mkdir -p $directory_of_run

for b in ${!position[@]}; do
    echo -e "pdf for ${position[$b]} \n"
    POSITION_STR=${POS_STRINGS[$b]}
    if [[ $test || "$set_cam" == "2" ]]; then
        echo "TEST"
    else
        cameras="$(ls -d $CSV_DIR/$POSITION_STR/[0-9]*[0-9]/ | sort -V )"
    fi

    for cam in $cameras; do
        camera=$(basename ${cam})
        echo $camera
        secff="$(ls -d $CSV_DIR/$POSITION_STR/${camera}/*.${camera}*/ | sort -V | head -1 | sed 's/.*1550\([^.]*\).*/\1/')"

        texheader="%\usepackage{etoolbox}
                    %% root folders: ---------------------
                    \extrafloats{1000}
                    \newcommand\rootserver{$rootserver}
                    \newcommand\rootcsv{$CSV_DIR}
                    \newcommand\rootrecord{$path_recordings}
                    \newcommand\plots{$PLOTS_TRAJECTORY}
                    \newcommand\block{$BLOCK}
                    \newcommand\posstr{$POSITION_STR}
                    \newcommand\programname{$PROGRAMNAME}
                    \newcommand\minindex{$MIN_BATCH_IDX}
                    \newcommand\maxindex{$MAX_BATCH_IDX}
                    \newcommand\subfigwidth{$SUBFIGURE_WIDTH}
                    \newcommand\subfigheight{$SUBFIGURE_HEIGHT}
                    \newcommand\files{$FILES}
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
        days="$(ls -d $CSV_DIR/$POSITION_STR/${camera}/*${STARTTIME}.${camera}*/ | sort -V )"

        for d in $days; do
            d=$(basename $d)
            day=${d: : 24}
            day_id=${day: : 15}
            daysarray="$daysarray\plotdayupdate{${day_id}}
            "
            # -----------
            # CSV Files
            # -----------
            filescsv="$(ls $CSV_DIR/$POSITION_STR/${camera}/*${STARTTIME}.${camera}*/${camera}_$day*.csv | sort -V)"
            C_i=0
            for f in $filescsv; do
                texheader="$texheader \setcsv{${day_id}$C_i}{\href{${PREFIX}${f}}{csv}}
                "
                let C_i++
            done
            # ---------
            # MP4 video files
            # ---------
            if [ $local ]; then
              skip=1
              #echo "--local is not reading mp4 file paths \n"
            else
              foldermp4="$(ls -d $path_recordings/${camera}/${day}*/ | head )"
              texheader="$texheader \addtext{$PREFIX$foldermp4}
              "
              filesmp4="$(ls $foldermp4/*.mp4 | sort -V)"
              C_i=0
              for f in $filesmp4; do
                  texheader="$texheader \setsub{${day_id}$C_i}{\href{$PREFIX$f}{mp4}}
                  "
                  let C_i++
              done
            fi
        done
        # daysarray=${daysarray%?}
        echo "${daysarray}" > $FILES/days_array.tex
        echo "$texheader" > $FILES/arrayoflinks.tex
        if [ $feeding ]; then
            echo "\input{$FILES/${BLOCK}feedingtime.tex}" >> $FILES/arrayoflinks.tex
        fi
        # run pdflatex two times
        END=2
        for k in $(seq 1 $END); do
            #pdflatex "\newcommand\secfirstplot{$secff}\newcommand\position{${position[$b]}}\newcommand\camera{${camera}}\input{main}"
            pdflatex --interaction=nonstopmode "\newcommand\arrayoflinks{${FILES}/arrayoflinks}\newcommand\secfirstplot{$secff}\newcommand\position{${position[$b]}}\newcommand\camera{${camera}}\input{main}" > log_tex.txt
            #pdflatex "\newcommand\secfirstplot{$secff}\newcommand\position{${position[$b]}}\newcommand\camera{${camera}}\input{main}"
        done
        mv main.pdf ${directory_of_run}/${PROGRAMNAME}_${BLOCK}_${camera}_${position[$b]}.pdf
    done
done

rm main.out main.log main.aux
echo "Execution time: $SECONDS seconds"
