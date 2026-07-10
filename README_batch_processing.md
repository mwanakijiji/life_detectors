# Batch Processing for Life Detectors

This document explains how to run batch jobs that change the `n_int` value in the INI configuration file and save results as FITS files with different names.

## Overview

The batch processing system allows you to:
- Run multiple calculations with different `n_int` (number of integrations) values
- Automatically save each result as a separate FITS file with a descriptive name
- Process multiple parameter combinations efficiently
- Generate plots and logs for each calculation

## Quick Start

### Command Line Usage

The simplest way to run a batch job is using the command line:

```bash
python batch_process.py --config modules/config/demo_config.ini \
                       --n-int 1000 3000 6000 9000 \
                       --output-dir batch_output \
                       --sources star exoplanet_model_10pc exozodiacal zodiacal
```

This will:
- Use the configuration file `modules/config/demo_config.ini`
- Run calculations with `n_int` values of 1000, 3000, 6000, and 9000
- Save results in the `batch_output` directory
- Include the specified astrophysical sources
- Create FITS files named `s2n_n1000.fits`, `s2n_n3000.fits`, etc.

### Python Script Usage

You can also use the batch processing functions in your own Python scripts:

```python
from batch_process import batch_process, run_single_calculation

# Run multiple calculations
results = batch_process(
    config_path="modules/config/demo_config.ini",
    n_int_values=[1000, 3000, 6000, 9000],
    output_dir="my_batch_output",
    sources_to_include=["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"],
    base_filename="my_s2n",
    overwrite=True,
    plot=False
)

# Run a single calculation
success = run_single_calculation(
    config_path="modules/config/demo_config.ini",
    sources_to_include=["star", "exoplanet_model_10pc"],
    n_int=5000,
    output_path="single_output/s2n_n5000.fits",
    overwrite=True,
    plot=False
)
```

## Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to the base configuration file | Required |
| `--n-int` | `-n` | List of n_int values to process | Required |
| `--output-dir` | `-o` | Output directory for FITS files | Required |
| `--sources` | `-s` | Sources to include in calculations | `["star", "exoplanet_model_10pc", "exozodiacal", "zodiacal"]` |
| `--base-filename` | `-f` | Base filename for output files | `"s2n"` |
| `--overwrite` | | Overwrite existing files | `True` |
| `--plot` | | Generate plots | `False` |
| `--verbose` | `-v` | Verbose output | `False` |

## Examples

### Example 1: Simple Parameter Sweep

```bash
python batch_process.py \
    --config modules/config/demo_config.ini \
    --n-int 1000 2000 3000 4000 5000 \
    --output-dir parameter_sweep \
    --base-filename s2n_sweep
```

### Example 2: Different Source Combinations

```bash
# Only planet and star
python batch_process.py \
    --config modules/config/demo_config.ini \
    --n-int 3000 6000 \
    --output-dir planet_only \
    --sources star exoplanet_model_10pc \
    --base-filename s2n_planet

# Only background sources
python batch_process.py \
    --config modules/config/demo_config.ini \
    --n-int 3000 6000 \
    --output-dir background_only \
    --sources star exozodiacal zodiacal \
    --base-filename s2n_background
```

### Example 3: With Plots and Verbose Output

```bash
python batch_process.py \
    --config modules/config/demo_config.ini \
    --n-int 1000 5000 \
    --output-dir with_plots \
    --plot \
    --verbose
```

## Output Files

The batch processor creates:

1. **FITS files**: Each calculation produces a FITS file with the S/N data
   - Filename format: `{base_filename}_n{n_int}.fits`
   - Example: `s2n_n3000.fits`
   - Contains the S/N array and configuration parameters in the header

2. **Log files**: Detailed logs for each calculation
   - Location: `logs/` directory
   - Format: `detectorsim_YYYYMMDD_HHMMSS.log`

3. **Plot files** (if `--plot` is used):
   - Various plots showing spectra, noise contributions, etc.
   - Saved in the same directory as the FITS files

## Configuration File Modifications

The batch processor automatically creates temporary configuration files with modified values:

- **`n_int`**: Changed to the specified value for each calculation
- **`save_s2n_data`**: Set to the output FITS file path
- All other parameters remain unchanged from the base configuration

## Available Sources

The following astrophysical sources can be included in calculations:

- `star`: Stellar spectrum
- `exoplanet_bb`: Exoplanet blackbody spectrum
- `exoplanet_model_10pc`: Exoplanet model spectrum (10 pc distance)
- `exozodiacal`: Exozodiacal dust emission
- `zodiacal`: Zodiacal dust emission

## Error Handling

The batch processor includes robust error handling:

- Individual calculation failures don't stop the entire batch
- Detailed error messages are logged
- Temporary configuration files are cleaned up automatically
- Success/failure status is reported for each calculation

## Performance Tips

1. **Disable plotting** for large batch jobs to save time and disk space
2. **Use appropriate n_int ranges** - very large values will take longer to compute
3. **Monitor disk space** - FITS files and plots can be large
4. **Check logs** if calculations fail to understand the cause

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   - Ensure the path to the config file is correct
   - Use absolute paths if relative paths don't work

2. **Output directory permissions**
   - Ensure you have write permissions to the output directory
   - The script will create the directory if it doesn't exist

3. **Memory issues with large n_int values**
   - Reduce the number of simultaneous calculations
   - Use smaller n_int values for testing

4. **FITS file already exists**
   - Use `--overwrite` flag to replace existing files
   - Or choose a different output directory

### Getting Help

Run with `--verbose` to see detailed output:

```bash
python batch_process.py --config your_config.ini --n-int 1000 --output-dir test --verbose
```

Check the log files in the `logs/` directory for detailed error information.
