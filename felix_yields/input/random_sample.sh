#!/usr/bin/env bash

# Usage:
# ./random_sample.sh input.txt output.txt 20

input_file="$1"
output_file="$2"
sample_size="$3"

# Clear output file
: > "$output_file"

# Copy first two lines exactly
head -n 2 "$input_file" >> "$output_file"

# Randomly sample from the remaining lines
tail -n +3 "$input_file" | shuf -n "$sample_size" >> "$output_file"
