# Life Detectors - Infrared Detector Noise Calculator

A Python package for calculating total noise in infrared detectors on telescopes observing stars.

## Features

- **Astrophysical Noise Sources**: Star, exoplanet, exozodiacal disk, zodiacal disk, and other sources
- **Instrumental Noise Sources**: Dark current, read noise, and other instrumental effects
- **Comprehensive S/N Calculations**: Convert all noise sources to ADU/pixel and calculate net signal-to-noise
- **Configurable**: YAML configuration files for easy parameter management
- **Extensible**: Designed for future GUI frontend and 2D array support

## Installation

```bash
pip install -e .
```

## Usage

### Command Line Interface

```bash
python -m modules.cli --config config.yaml --output results.json
```

### Python API

```python
from modules.core import NoiseCalculator
from modules.config import load_config

config = load_config("config.yaml")
calculator = NoiseCalculator(config)
results = calculator.calculate_snr()
```

## Package Structure

```
modules/
├── core/                    # Core calculation modules
│   ├── __init__.py
│   ├── astrophysical.py    # Astrophysical noise sources
│   ├── instrumental.py     # Instrumental noise sources
│   ├── conversions.py      # Unit conversions
│   └── calculator.py       # Main calculation engine
├── config/                 # Configuration management
│   ├── __init__.py
│   ├── loader.py          # Config file loading
│   └── validator.py       # Config validation
├── data/                  # Data handling
│   ├── __init__.py
│   ├── spectra.py         # Spectral data loading
│   └── units.py          # Unit definitions and conversions
├── cli/                   # Command line interface
│   ├── __init__.py
│   └── main.py           # CLI entry point
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── helpers.py        # Helper functions
├── tests/                 # Unit tests
│   ├── __init__.py
│   ├── test_astrophysical.py
│   ├── test_instrumental.py
│   ├── test_conversions.py
│   ├── test_calculator.py
│   └── test_config.py
├── examples/              # Example configurations
│   ├── basic_config.yaml
│   └── advanced_config.yaml
├── docs/                  # Documentation
│   └── uml/              # UML diagrams
├── setup.py              # Package setup
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Configuration

The package uses YAML configuration files to define all parameters:

```yaml
# Example configuration
telescope:
  collecting_area: 25.0  # m^2
  plate_scale: 0.1      # arcsec/pixel
  throughput: 0.8       # dimensionless

target:
  distance: 10.0        # parsecs
  nulling_factor: 0.01  # dimensionless

detector:
  read_noise: 5.0       # e-/pixel
  dark_current: 0.1     # e-/pixel/sec
  gain: 2.0             # e-/ADU
  integration_time: 3600 # seconds

astrophysical_sources:
  star:
    spectrum_file: "data/star_spectrum.txt"
  exoplanet:
    spectrum_file: "data/exoplanet_spectrum.txt"
  exozodiacal:
    spectrum_file: "data/exozodiacal_spectrum.txt"
  zodiacal:
    spectrum_file: "data/zodiacal_spectrum.txt"
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building Documentation

```bash
cd docs && make html
```

## Future Extensions

- GUI frontend using tkinter or web interface
- 2D array support for resolved objects
- Transmission maps over field of view