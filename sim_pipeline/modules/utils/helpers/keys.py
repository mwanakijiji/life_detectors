"""Float and HDF5 key helpers."""

import re
from pathlib import Path
from typing import Optional, Union

def parse_sky_position_arcsec_yx(pos_str: str) -> tuple[float, float]:
    """Parse on-sky position string as y, x offsets in arcsec (y first, x second)."""
    parts = [float(v.strip()) for v in str(pos_str).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated values (y, x arcsec), got {pos_str!r}")
    return parts[0], parts[1]


ANGLE_FILENAME_DECIMALS = 2
DC_GROUP_DECIMALS = 3
QE_GROUP_DECIMALS = 2

_ANGLE_HDF5_STEM_RE = re.compile(r"^angle_(?P<angle>[0-9.]+)(?:_qe_|$)")
_DC_QE_GROUP_RE = re.compile(r"^dc_(?P<dc>.+)_qe_(?P<qe>[0-9.]+)$")


def canonical_angle_deg(angle_deg: float) -> float:
    """Match angle keys to HDF5 filename precision (``angle_{:06.2f}``)."""
    return round(float(angle_deg), ANGLE_FILENAME_DECIMALS)


def canonical_dc_rate(dc_rate: float) -> float:
    """Match dark-current keys to HDF5 group precision (``dc_{:06.3f}``)."""
    return round(float(dc_rate), DC_GROUP_DECIMALS)


def canonical_qe(qe: float) -> float:
    """Match QE keys to HDF5 group precision (``qe_{:04.2f}``)."""
    return round(float(qe), QE_GROUP_DECIMALS)


def format_angle_qe_hdf5_name(angle_deg: float, qe: float) -> str:
    """Return ``angle_{angle}_qe_{qe}.hdf5`` using the same formatting as writes."""
    return (
        f"angle_{canonical_angle_deg(angle_deg):06.2f}"
        f"_qe_{canonical_qe(qe):04.2f}.hdf5"
    )


def parse_angle_from_hdf5_path(path: Union[str, Path]) -> float:
    """Parse the angle embedded in an ``angle_*.hdf5`` filename."""
    stem = Path(path).stem
    match = _ANGLE_HDF5_STEM_RE.match(stem)
    if match is None:
        raise ValueError(f"Cannot parse angle from HDF5 filename: {path!r}")
    return canonical_angle_deg(float(match.group("angle")))


def parse_qe_from_hdf5_path(path: Union[str, Path]) -> Optional[float]:
    """Parse QE from ``angle_*_qe_{qe}.hdf5``; return None if the name has no QE suffix."""
    stem = Path(path).stem
    if "_qe_" not in stem:
        return None
    qe_str = stem.rsplit("_qe_", 1)[-1]
    return canonical_qe(float(qe_str))


def hdf5_path_matches_qe(path: Union[str, Path], qe: float) -> bool:
    """True when ``path`` is an angle HDF5 for ``qe`` (legacy names without QE always match)."""
    file_qe = parse_qe_from_hdf5_path(path)
    if file_qe is None:
        return True
    return file_qe == canonical_qe(qe)


def parse_dc_qe_group(dc_qe_str: str) -> tuple[float, float]:
    """Parse ``dc_{dc}_qe_{qe}`` group names written by ``record_info_at_angle_and_qe``."""
    match = _DC_QE_GROUP_RE.match(dc_qe_str)
    if match is None:
        raise ValueError(f"Unrecognized HDF5 group name: {dc_qe_str!r}")
    return canonical_dc_rate(float(match.group("dc"))), canonical_qe(float(match.group("qe")))


def lookup_float_key(mapping: dict, key: float, *, label: str = "key", decimals: int = 6):
    """Dict lookup tolerant of tiny float formatting differences."""
    resolved = resolve_float_key(mapping, key, label=label, decimals=decimals)
    return mapping[resolved]


def resolve_float_key(mapping: dict, key: float, *, label: str = "key", decimals: int = 6) -> float:
    """Return the mapping key that matches ``key`` within formatting tolerance."""
    key = float(key)
    if key in mapping:
        return key
    tol = 10 ** (-decimals)
    for existing_key in mapping:
        if abs(float(existing_key) - key) <= tol:
            return float(existing_key)
    available = ", ".join(f"{existing_key!r}" for existing_key in sorted(mapping))
    raise KeyError(f"{label} {key!r} not found; available: {available}")
