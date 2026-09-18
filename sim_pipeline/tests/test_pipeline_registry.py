"""Tests for the decorator-based pipeline stage registry."""

from modules.pipeline_registry import (
    ensure_stages_registered,
    graph_from_registry,
)


EXPECTED_NODES = {
    "calculate_received_astro_fluxes",
    "arrange_onsky_scene",
    "generate_instrument_transmission",
    "pass_through_transmission_screens",
    "pass_through_aperture",
    "pass_from_aperture_to_detector",
    "disperse_astro_signals_on_detector",
    "apply_detector_effects",
    "calculate_instrinsic_instrumental_noise",
    "combine_astro_and_instrum_signals",
    "chop_signal",
    "record_info_at_angle_and_qe",
}

EXPECTED_EDGES = {
    ("calculate_received_astro_fluxes", "arrange_onsky_scene"),
    ("arrange_onsky_scene", "pass_through_transmission_screens"),
    ("generate_instrument_transmission", "pass_through_transmission_screens"),
    ("pass_through_transmission_screens", "pass_through_aperture"),
    ("pass_through_aperture", "pass_from_aperture_to_detector"),
    ("pass_from_aperture_to_detector", "disperse_astro_signals_on_detector"),
    ("disperse_astro_signals_on_detector", "apply_detector_effects"),
    ("apply_detector_effects", "combine_astro_and_instrum_signals"),
    ("pass_through_aperture", "calculate_instrinsic_instrumental_noise"),
    ("calculate_instrinsic_instrumental_noise", "combine_astro_and_instrum_signals"),
    ("combine_astro_and_instrum_signals", "chop_signal"),
    ("chop_signal", "record_info_at_angle_and_qe"),
}


def test_registry_includes_expected_stages():
    registry = ensure_stages_registered()
    assert set(registry) == EXPECTED_NODES
    assert registry["apply_detector_effects"].planned is True


def test_graph_edges_match_pipeline_dag():
    nodes, edges, clusters, planned, stages = graph_from_registry()
    assert set(nodes) == EXPECTED_NODES
    assert set(edges) == EXPECTED_EDGES
    assert planned == frozenset({"apply_detector_effects"})
    assert stages == list(nodes.values())
    assert "astrophysics" in {c["label"] for c in clusters.values()}
