#!/usr/bin/env bash

# Usage:
# ./sample_lines.sh input.txt output.txt 20
# ./sample_lines.sh input.txt output.txt 20 ' A '
# where 20 is the number of lines to sample, and ' A ' is a string filter for the lines to sample.

input_file="$1"
output_file="$2"
sample_size="$3"
filter="$4"

# Clear output file
: > "$output_file"

# Copy first two lines exactly
head -n 2 "$input_file" >> "$output_file"

# Randomly sample from the remaining lines (optionally filtered)
if [[ -n "$filter" ]]; then
  tail -n +3 "$input_file" | grep -F "$filter" | shuf -n "$sample_size" >> "$output_file"
else
  tail -n +3 "$input_file" | shuf -n "$sample_size" >> "$output_file"
fi
