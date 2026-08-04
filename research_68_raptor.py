"""Import shim for the hyphenated RAPTOR research directory."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location(
    "research_68_raptor.raptor", Path(__file__).parent / "research/68-raptor-spike/raptor.py"
)
if _spec is None or _spec.loader is None:
    raise ImportError("could not load RAPTOR prototype")
raptor = module_from_spec(_spec)
sys.modules[_spec.name] = raptor
_spec.loader.exec_module(raptor)
