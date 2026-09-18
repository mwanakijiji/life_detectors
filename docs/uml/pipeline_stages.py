"""Pipeline stage graph for docs/uml flow diagrams.

Built from ``@pipeline_stage`` annotations on the real callables
(see ``sim_pipeline.modules.pipeline_registry``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SIM_PIPELINE_ROOT = Path(__file__).resolve().parents[2] / "sim_pipeline"
if str(_SIM_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIM_PIPELINE_ROOT))

from modules.pipeline_registry import graph_from_registry  # noqa: E402

NODES, EDGES, CLUSTERS, PLANNED_STAGES, STAGES = graph_from_registry()

__all__ = [
    "CLUSTERS",
    "EDGES",
    "NODES",
    "PLANNED_STAGES",
    "STAGES",
]
