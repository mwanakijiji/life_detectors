start_time=$(date +%s)
echo "Batch job started at: $(date)"

for file in output_psg_cfg_*.txt; do
  curl --data-urlencode "file@${file}" https://psg.gsfc.nasa.gov/api.php \
    -o "${file%.txt}.response"
done

end_time=$(date +%s)
echo "Batch job completed at: $(date)"
echo "Batch job took: $((end_time - start_time)) seconds"


