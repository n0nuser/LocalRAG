"""Import shim for the hyphenated GraphRAG research directory."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location(
    "research_67_graphrag.graphrag",
    Path(__file__).parent / "research/67-graphrag-spike/graphrag.py",
)
if _spec is None or _spec.loader is None:
    raise ImportError("could not load GraphRAG prototype")
graphrag = module_from_spec(_spec)
sys.modules[_spec.name] = graphrag
_spec.loader.exec_module(graphrag)
