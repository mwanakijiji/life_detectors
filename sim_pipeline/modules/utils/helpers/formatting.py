"""Plot title and label formatting helpers."""

import configparser
from typing import Optional

from ..loader import config_getboolean

def _config_get(config, section: str, key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if isinstance(config, configparser.ConfigParser):
            if not config.has_section(section):
                return default
            return config[section].get(key, default)
        if isinstance(config, dict):
            return config.get(section, {}).get(key, default)
    except Exception:
        return default
    return default

def _config_set_plot_title_context(config, value: str) -> None:
    if not value:
        return
    if isinstance(config, configparser.ConfigParser):
        if not config.has_section("plotting"):
            config.add_section("plotting")
        config.set("plotting", "title_context", value)
    elif isinstance(config, dict):
        config.setdefault("plotting", {})["title_context"] = value


def _get_plot_title_context(config) -> str:
    if isinstance(config, configparser.ConfigParser):
        if config.has_section("plotting"):
            return config["plotting"].get("title_context", "").strip()
        return ""
    if isinstance(config, dict):
        return str(config.get("plotting", {}).get("title_context", "")).strip()
    return ""


def build_system_params_title(config) -> str:
    lines = []

    collecting_area = _config_get(config, "telescope", "collecting_area")
    if collecting_area is not None:
        lines.append(f"collecting area = {float(collecting_area):.2f} m^2")

    eta_t = _config_get(config, "telescope", "eta_t")
    if eta_t is not None:
        lines.append(f"telescope throughput = {float(eta_t):.2f}")

    nulling = _config_get(config, "nulling", "null")
    nulling_factor = _config_get(config, "nulling", "nulling_factor")
    if nulling is not None and nulling_factor is not None:
        lines.append(
            f"stellar nulling = {bool(nulling)}, nulling transmission = {float(nulling_factor):.1e}"
        )

    lambda_rel = _config_get(config, "target", "lambda_rel_lon_los")
    beta = _config_get(config, "target", "beta_lat_los")
    if lambda_rel is not None and beta is not None:
        lines.append(
            fr"galactic $\lambda_{{\rm rel}}$ = {float(lambda_rel):.2f} deg, $\beta$ = {float(beta):.2f} deg"
        )

    z_exozodiacal = _config_get(config, "target", "z_exozodiacal")
    if z_exozodiacal is not None:
        lines.append(f"z_exozodiacal = {float(z_exozodiacal)}")

    A_albedo = _config_get(config, "target", "A_albedo")
    if A_albedo is not None:
        lines.append(f"A_albedo = {float(A_albedo)}")

    L_star = _config_get(config, "target", "L_star")
    if L_star is not None:
        lines.append(f"L_star = {float(L_star)} L_sol")

    rad_star = _config_get(config, "target", "rad_star")
    if rad_star is not None:
        lines.append(f"rad_star = {float(rad_star)} solar radii")

    T_star = _config_get(config, "target", "T_star")
    if T_star is not None:
        lines.append(f"T_star = {float(T_star)} K")

    rad_planet = _config_get(config, "target", "rad_planet")
    if rad_planet is not None:
        lines.append(f"rad_planet = {float(rad_planet)} Earth radii")

    pl_temp = _config_get(config, "target", "pl_temp")
    if pl_temp is not None:
        lines.append(f"pl_temp = {float(pl_temp)} K")

    distance = _config_get(config, "target", "distance")
    if distance is not None:
        lines.append(f"distance = {float(distance)} pc")

    return "\n".join(lines)


def build_observation_detector_title(config) -> str:
    lines = []

    t_int_frame = _config_get(config, "observation", "t_int_frame")
    if t_int_frame is not None:
        lines.append(f"t_int_frame = {float(t_int_frame)} s")

    n_angles = _config_get(config, "observation", "N_angles")
    if n_angles is not None:
        lines.append(f"N_angles = {int(float(n_angles))}")

    n_int_per_angle = _config_get(config, "observation", "N_int_per_angle")
    if n_int_per_angle is not None:
        lines.append(f"N_int_per_angle = {int(float(n_int_per_angle))}")

    t_int_total = float(t_int_frame) * float(n_angles) * float(n_int_per_angle)
    if t_int_total is not None:
        lines.append(f"t_int_total = {float(t_int_total):.0f} s")

    qe = _config_get(config, "detector", "quantum_efficiency")

    if qe is not None:
        lines.append(f"QE = {float(config['detector']['quantum_efficiency'])}")

    dark_current_sweep = _config_get(config, "detector", "dark_current")
    if dark_current_sweep is not None:
        lines.append(f"dark current sweep = {dark_current_sweep} e-/pix/s")

    read_noise = _config_get(config, "detector", "read_noise")
    if read_noise is not None:
        lines.append(f"read noise = {read_noise} e- rms")

    gain = _config_get(config, "detector", "gain")
    if gain is not None:
        lines.append(f"gain = {float(gain)} e-/ADU")

    spec_res = _config_get(config, "detector", "spec_res")
    if spec_res is not None:
        lines.append(f"spec_res R = {float(spec_res)}")


    return "\n".join(lines)


def ensure_plot_title_context(config) -> str:
    existing = _get_plot_title_context(config)
    if existing:
        return existing
    parts = [build_system_params_title(config), build_observation_detector_title(config)]
    built = "\n".join(part for part in parts if part)
    _config_set_plot_title_context(config, built)
    return built


def _config_section_keys(config, section: str) -> list[str]:
    if isinstance(config, configparser.ConfigParser):
        if not config.has_section(section):
            return []
        return list(config[section].keys())
    if isinstance(config, dict):
        return list(config.get(section, {}).keys())
    return []


def build_astrophysical_sources_to_use_title(config) -> str:
    """Format [astrophysical_sources_to_use] flags for plot title side column."""
    keys = _config_section_keys(config, "astrophysical_sources_to_use")
    if not keys:
        return ""

    lines = ["astrophysical sources:"]
    for key in keys:
        enabled = config_getboolean(config, "astrophysical_sources_to_use", key)
        lines.append(f"  {key} = {enabled}")
    return "\n".join(lines)


def _join_two_column_text(left: str, right: str, gap: int = 4) -> str:
    """Place two newline-separated blocks side by side."""
    left_lines = left.splitlines() if left else []
    right_lines = right.splitlines() if right else []
    if not left_lines:
        return "\n".join(right_lines)
    if not right_lines:
        return "\n".join(left_lines)

    left_width = max(len(line) for line in left_lines)
    n_rows = max(len(left_lines), len(right_lines))
    rows = []
    for row_idx in range(n_rows):
        left_cell = left_lines[row_idx] if row_idx < len(left_lines) else ""
        right_cell = right_lines[row_idx] if row_idx < len(right_lines) else ""
        rows.append(f"{left_cell:<{left_width}}{' ' * gap}{right_cell}")
    return "\n".join(rows)


def format_plot_title(base_title: str, config, font_size: int = 12) -> str:
    title_context = _get_plot_title_context(config)
    sources_context = build_astrophysical_sources_to_use_title(config)
    body = _join_two_column_text(title_context, sources_context)
    if not body:
        return base_title
    separator = "\n" if base_title else ""
    # If 'base_title' is present, underline it; otherwise, just return the context.
    if base_title:
        # Underline the base title with '=' (length matches without ANSI/formatting, just plain text)
        underline = "=" * len(base_title)
        return f"{base_title}\n{underline}{separator}{body}"
    else:
        return f"{body}"

# end bunch of functions to get and set the plot title context
########################################################


def format_astro_source_label(source_name: str) -> str:
    """Map internal astrophysical source keys to plot legend labels."""
    if source_name == "star" or source_name.startswith("star_"):
        return "Star"
    if source_name.startswith("exozodiacal"):
        return "Exozodiacal"
    if source_name.startswith("exoplanet"):
        return "Exoplanet"
    if source_name == "zodiacal":
        return "Zodiacal"
    return source_name
