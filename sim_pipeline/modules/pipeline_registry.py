"""
Lightweight pipeline-stage registry (Option 2).

Decorate the real callables with ``@pipeline_stage(...)``; docs tooling walks
the registry to build the flowchart. ``batch_process`` keeps its own control
flow — this module does not orchestrate execution.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

# any callable; the decorator returns it unchanged, so its signature is kept
F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class StageSpec:
    """Metadata for one generate-sims pipeline stage."""

    id: str
    label: str
    depends_on: Tuple[str, ...] = ()
    cluster: Optional[str] = None
    cluster_color: Optional[str] = None
    planned: bool = False
    qualname: Optional[str] = None


_REGISTRY: Dict[str, StageSpec] = {}
# mutable module state (avoids `global`): whether the stage modules were imported
_STATE = {"imports_done": False}

# modules whose @pipeline_stage decorators populate the registry when imported;
# import order is unimportant
_STAGE_MODULES = (
    "modules.core.astrophysical",
    "modules.core.calculator.extracted_cube",
    "modules.core.instrumental.aperture",
    "modules.core.instrumental.dispersion",
    "modules.core.instrumental.effects",
    "modules.core.instrumental.readouts",
    "modules.core.instrumental.transmission",
    "modules.utils.helpers.hdf5_io",
)


def pipeline_stage(
    *,
    name: Optional[str] = None,
    depends_on: Sequence[str] = (),
    cluster: Optional[str] = None,
    cluster_color: Optional[str] = None,
    planned: bool = False,
) -> Callable[[F], F]:
    """
    Register a callable as a pipeline stage for docs / discovery.

    ``depends_on`` lists immediate predecessor stage ids (DAG parents).
    """

    def decorator(fn: F) -> F:
        stage_id = name or fn.__name__
        _REGISTRY[stage_id] = StageSpec(
            id=stage_id,
            label=stage_id,
            depends_on=tuple(depends_on),
            cluster=cluster,
            cluster_color=cluster_color,
            planned=planned,
            qualname=getattr(fn, "__qualname__", stage_id),
        )
        return fn

    return decorator


def register_planned_stage(
    stage_id: str,
    *,
    depends_on: Sequence[str] = (),
    cluster: Optional[str] = None,
    cluster_color: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    """Register a stage that is not yet implemented as a real callable."""
    _REGISTRY[stage_id] = StageSpec(
        id=stage_id,
        label=label or stage_id,
        depends_on=tuple(depends_on),
        cluster=cluster,
        cluster_color=cluster_color,
        planned=True,
        qualname=None,
    )


def get_registry() -> Dict[str, StageSpec]:
    """Return a copy of the registered stages, keyed by stage id."""
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Test helper."""
    _REGISTRY.clear()
    _STATE["imports_done"] = False


def ensure_stages_registered() -> Dict[str, StageSpec]:
    """Import decorated modules so the registry is populated."""
    if not _STATE["imports_done"]:
        for module_name in _STAGE_MODULES:
            importlib.import_module(module_name)
        _STATE["imports_done"] = True
    return get_registry()


def graph_from_registry() -> Tuple[
    Dict[str, str],
    List[Tuple[str, str]],
    Dict[str, Dict],
    frozenset,
    List[str],
]:
    """
    Build NODES, EDGES, CLUSTERS, PLANNED_STAGES, STAGES from the registry.

    Edges are ``(predecessor, stage)`` for each ``depends_on`` entry.
    """
    registry = ensure_stages_registered()
    nodes = {stage_id: spec.label for stage_id, spec in registry.items()}
    edges: List[Tuple[str, str]] = []
    for stage_id, spec in registry.items():
        for parent in spec.depends_on:
            if parent not in registry:
                raise KeyError(
                    f"Stage {stage_id!r} depends on unknown stage {parent!r}"
                )
            edges.append((parent, stage_id))

    clusters: Dict[str, Dict] = {}
    for stage_id, spec in registry.items():
        if not spec.cluster:
            continue
        # Stable cluster key from label (slug)
        key = spec.cluster.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if key not in clusters:
            clusters[key] = {
                "label": spec.cluster,
                "color": spec.cluster_color or "#666666",
                "nodes": [],
            }
        clusters[key]["nodes"].append(stage_id)
        if spec.cluster_color:
            clusters[key]["color"] = spec.cluster_color

    planned = frozenset(s.id for s in registry.values() if s.planned)
    stages = list(nodes.values())
    return nodes, edges, clusters, planned, stages
