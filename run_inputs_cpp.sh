#!/bin/bash

input_dir="inputs"
output_dir="outputs"

mkdir -p "$output_dir"

# Loop through all input files in the inputs directory matching pattern input_group*
for input_file in "$input_dir"/input_group*; do
    filename=$(basename "$input_file")
    group_id="${filename#input_}"  # removes "input_"
    
    output_file="output_${group_id}"

    echo "Processing $input_file -> $output_dir/$output_file"
    
    ./algobowl "$input_file" > "$output_dir/$output_file"
done

echo "All files processed"

