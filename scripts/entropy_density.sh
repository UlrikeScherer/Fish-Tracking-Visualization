
#!/bin/bash
source fishproviz/config.env # get the following variables

cd tex
entropy_density="entropy_density"
directory_of_run=$path_csv_local/$VIS_DIR
PLOTS_TRAJECTORY=$path_csv_local/$PLOTS_DIR
FILES="$PLOTS_TRAJECTORY/$TEX_DIR"
mkdir $FILES
echo "
-------------------------
START generating $entropy_density PDFs for: 
$PROJECT_ID
-------------------------"

mkdir -p $entropy_density
texheader="
    \newcounter{cntfish}
    \newcommand\fishlist{}
    \newcommand\setfish[2]{%
    \csdef{fish#1}{#2}}
    \newcommand\addfish[1]{%
    \csdef{fish\thecntfish}{#1}%
    \stepcounter{cntfish}}
    \newcommand\getfish[1]{%
    \csuse{fish#1}}"

unique_base_names() {
  local path=$directory_of_run/$entropy_density

  # List all files in the specified path with the given pattern
  for file in $path/*_overall*.png; do
    # Extract the base name of each file
    base_name="$(basename "$file")"
    IFS="_" read -ra elements <<< "$base_name"
    # Join the first two elements back together
    output_string="${elements[0]}_${elements[1]}"
    echo "$output_string"
  done | sort -u
}


for fish in $(unique_base_names); do
    echo "fish: $fish"
    texheader="$texheader
    \addfish{$fish}"
done

echo "$texheader" > $FILES/"fish_keys.tex"

END=2
for k in $(seq 1 $END); do 
    pdflatex --interaction=nonstopmode "\newcommand\plots{$path_csv_local/$VIS_DIR}\newcommand\block{$PROJECT_ID}\newcommand\files{$FILES}\input{$entropy_density}"
done
mv ${entropy_density}.pdf $directory_of_run/$entropy_density/${entropy_density}_sum.pdf

rm ${entropy_density}.out ${entropy_density}.log ${entropy_density}.aux

echo "DONE!"
echo "$(unique_base_names)"