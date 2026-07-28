"""Config path and sweep helpers."""

import configparser
import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy.visualization import quantity_support

logger = logging.getLogger(__name__)

def get_sweep_range(obs: dict, prefix: str) -> list[float]:
    """
    Build [start, start+step, ..., stop] from obs[prefix_start], obs[prefix_stop], obs[prefix_step].

    The stop value is included by extending the range by one step.
    """
    start_ = float(obs[f'{prefix}_start'])
    stop_ = float(obs[f'{prefix}_stop'])
    step_ = float(obs[f'{prefix}_step'])

    return np.arange(start_, stop_, step_).tolist()


def validate_file_path(filepath: Union[str, Path]) -> bool:
    """
    Validate that a file path exists and is readable.
    
    Args:
        filepath: Path to validate
        
    Returns:
        True if file exists and is readable
    """
    try:
        path = Path(filepath)
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False
def _normalize_output_root(output_root: str) -> str:
    """Return an absolute output directory with a trailing separator."""
    normalized = str(Path(output_root).expanduser().resolve())
    return normalized if normalized.endswith(os.sep) else normalized + os.sep

def enable_plot_units():
    """Let matplotlib label axes from astropy Quantity units."""
    quantity_support()

def apply_output_root_override(config_path: str, output_root: Optional[str]) -> str:
    """
    Override the batch output root in a config file.

    The file is updated in place so downstream temporary config generation
    inherits the run-specific root directory.
    """
    if not output_root:
        return config_path

    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.has_section("dirs"):
        config.add_section("dirs")

    normalized_root = _normalize_output_root(output_root)
    os.makedirs(normalized_root, exist_ok=True)
    config.set("dirs", "save_s2n_data_unique_dir", normalized_root)

    with open(config_path, "w") as f:
        config.write(f)

    logger.info(f"Using overridden output root: {normalized_root}")
    return config_path


def modify_config_file_sweep(config_path: str, qe: float, run_id: Optional[str] = None) -> str:
    """
    Create a modified configuration file with new quantum efficiency value.

    Returns:
        Path to the temporary modified configuration file.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    config.set('detector', 'quantum_efficiency', str(qe))

    temp_config_dir = os.path.dirname(config_path) + '/parameter_sweeps/'
    qe_str = f"{qe:04.2f}".replace('.', 'p')
    run_suffix = f"_{run_id}" if run_id else ""
    temp_config_path = temp_config_dir + os.path.basename(config_path).replace(
        '.ini', f'_temp_qe{qe_str}{run_suffix}.ini'
    )
    if not os.path.exists(temp_config_dir):
        os.makedirs(temp_config_dir, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        config.write(f)

    return temp_config_path


def modify_config_file_pl_system_params(
    config_path: str,
    base_filename: str,
    system_params: dict,
    lum_types: dict,
    run_id: Optional[str] = None,
) -> str:
    """
    Create a modified config for one planet from a population model.

    Returns:
        Path to the temporary modified configuration file.
    """
    if system_params is None:
        return config_path

    config = configparser.ConfigParser()
    config.read(config_path)

    config.set('target', 'distance', str(system_params['Ds']))
    config.set('target', 'rad_planet', str(system_params['Rp']))
    config.set('target', 'pl_temp', str(system_params['Tp']))
    config.set('target', 'rad_star', str(system_params['Rs']))
    config.set('target', 't_star', str(system_params['Ts']))
    config.set('target', 'z_exozodiacal', str(system_params['z']))
    config.set('target', 'lambda_rel_lon_los', str(system_params['eclip_lon']))
    config.set('target', 'beta_lat_los', str(system_params['eclip_lat']))
    config.set('target', 'L_star', str(lum_types[system_params['Stype'].lower()]))
    config.set('target', 'psg_spectrum_file_name', str(system_params['abs_file_name_psg_spectrum']))
    logger.info(f"NASA PSG spectrum file name: {system_params['abs_file_name_psg_spectrum']}")
    config.set('target', 'Stype', str(system_params['Stype']))
    config.set('target', 'Nuniverse', str(system_params['Nuniverse']))
    config.set('target', 'Nstar', str(system_params['Nstar']))

    nuniverse_part = f"Nuniverse_{config['target']['Nuniverse']}"
    nstar_part = f"Nstar_{config['target']['Nstar']}"
    dist_part = f"dist_{config['target']['distance']}"
    rp_part = f"Rp_{config['target']['rad_planet']}"
    rs_part = f"Rs_{config['target']['rad_star']}"
    ts_part = f"Ts_{config['target']['t_star']}"
    l_part = f"L_{config['target']['L_star']}"
    z_part = f"z_{config['target']['z_exozodiacal']}"
    eclip_lon_part = f"eclip_lon_{config['target']['lambda_rel_lon_los']}"
    eclip_lat_part = f"eclip_lat_{config['target']['beta_lat_los']}"
    stype_part = f"Stype_{config['target']['Stype']}"

    file_basename_string = (
        f"temp_{base_filename}_"
        f"{nuniverse_part}_"
        f"{nstar_part}_"
        f"{dist_part}_"
        f"{rp_part}_"
        f"{rs_part}_"
        f"{ts_part}_"
        f"{l_part}_"
        f"{z_part}_"
        f"{eclip_lon_part}_"
        f"{eclip_lat_part}_"
        f"{stype_part}"
    )

    config.set(
        'dirs',
        'save_s2n_data_unique_dir',
        config['dirs']['save_s2n_data_unique_dir'] + file_basename_string + '/',
    )

    run_suffix = f"_{run_id}" if run_id else ""
    temp_config_path = (
        str(config['dirs']['save_s2n_data_unique_dir']) + file_basename_string + f'{run_suffix}.ini'
    )

    temp_config_dir = os.path.dirname(temp_config_path)
    os.makedirs(temp_config_dir, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        config.write(f)
    logger.info(f"Created temporary config file for one planetary system: {temp_config_path}")

    return temp_config_path