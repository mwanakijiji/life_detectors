"""Tests for the per-run output directory tree (modules/utils/helpers/output_dirs.py)."""

# pytest fixtures are injected by argument name
# pylint: disable=redefined-outer-name

import configparser
import os
from datetime import datetime
from unittest.mock import patch

import pytest

from batch_process import batch_qe_nint_process
from modules.utils.helpers.output_dirs import (
    COUNTS_TIME,
    FYI,
    INTERMEDIATE,
    OUTPUT_SUBDIRS,
    SNR,
    assert_no_previous_outputs,
    make_run_root,
    output_dir,
    previous_outputs_for_qe,
    resolve_in_run_dir,
    run_stamp,
)

NOW = datetime(2026, 11, 5, 14, 30, 12)


def _config(root):
    return {"dirs": {"save_s2n_data_unique_dir": str(root)}}


def test_run_stamp_format():
    assert run_stamp(NOW) == "20261105143012"


def test_make_run_root_is_new_and_unique(tmp_path):
    first = make_run_root(str(tmp_path), now=NOW)
    second = make_run_root(str(tmp_path), now=NOW)  # same second
    assert first == str(tmp_path / "run_20261105143012") + os.sep
    assert second == str(tmp_path / "run_20261105143012_2") + os.sep
    assert os.path.isdir(first) and os.path.isdir(second)
    assert not os.listdir(first)


@pytest.mark.parametrize("kind", [FYI, INTERMEDIATE, COUNTS_TIME, SNR])
def test_output_dir_creates_subdirectory(tmp_path, kind):
    path = output_dir(_config(tmp_path), kind)
    assert path == str(tmp_path / OUTPUT_SUBDIRS[kind]) + os.sep
    assert os.path.isdir(path)


def test_output_dir_accepts_root_without_trailing_separator(tmp_path):
    root = str(tmp_path / "run")
    assert output_dir(_config(root), FYI).startswith(root + os.sep)


def test_output_dir_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        output_dir(_config(tmp_path), "nope")


def test_resolve_in_run_dir(tmp_path):
    config = _config(tmp_path)
    assert resolve_in_run_dir(config, "wavel_solution.fits", INTERMEDIATE) == str(
        tmp_path / "intermediate" / "wavel_solution.fits"
    )
    assert resolve_in_run_dir(config, "/abs/w.fits", INTERMEDIATE) == "/abs/w.fits"
    assert resolve_in_run_dir(config, "sub/w.fits", INTERMEDIATE) == "sub/w.fits"


def test_previous_outputs_detected_per_qe(tmp_path):
    config = _config(tmp_path)
    (tmp_path / "intermediate").mkdir()
    (tmp_path / "intermediate" / "angle_000.00_qe_0.60.hdf5").write_text("x")
    (tmp_path / "products_counts_vs_time").mkdir()
    (tmp_path / "products_counts_vs_time" / "extracted_cube_qe_0.60_dc_00.000.fits"
     ).write_text("x")

    assert len(previous_outputs_for_qe(str(tmp_path), 0.6)) == 2
    with pytest.raises(FileExistsError, match="refusing to mix runs"):
        assert_no_previous_outputs(config, 0.6)
    # another QE in the same run tree is fine
    assert_no_previous_outputs(config, 0.7)


def test_empty_tree_passes_check(tmp_path):
    assert_no_previous_outputs(_config(tmp_path), 0.6)


def _write_base_config(path, save_dir, generate_sims):
    config = configparser.ConfigParser()
    config["tasks"] = {"generate_sims": str(generate_sims)}
    config["dirs"] = {"save_s2n_data_unique_dir": str(save_dir) + "/"}
    with open(path, "w", encoding="utf-8") as f:
        config.write(f)
    return str(path)


@patch("batch_process.run_single_calculation", return_value=True)
def test_batch_shares_one_fresh_run_root_across_qe_values(mock_run, tmp_path):
    base = tmp_path / "save"
    config_path = _write_base_config(tmp_path / "c.ini", base, generate_sims=True)

    batch_qe_nint_process(
        base_config_path=config_path, qe_values=[0.5, 0.6], sources_to_include=[]
    )
    roots = {c.kwargs["output_root"] for c in mock_run.call_args_list}
    assert len(roots) == 1
    (root,) = roots
    assert os.path.basename(root.rstrip(os.sep)).startswith("run_")
    assert os.path.dirname(root.rstrip(os.sep)) == str(base)

    # a second batch gets a different, new tree
    batch_qe_nint_process(
        base_config_path=config_path, qe_values=[0.5], sources_to_include=[]
    )
    assert mock_run.call_args.kwargs["output_root"] != root


@patch("batch_process.run_single_calculation", return_value=True)
def test_no_new_run_root_when_not_generating_sims(mock_run, tmp_path):
    base = tmp_path / "save"
    config_path = _write_base_config(tmp_path / "c.ini", base, generate_sims=False)
    batch_qe_nint_process(
        base_config_path=config_path, qe_values=[0.5], sources_to_include=[]
    )
    assert mock_run.call_args.kwargs["output_root"] is None
    assert not base.exists() or not os.listdir(base)
