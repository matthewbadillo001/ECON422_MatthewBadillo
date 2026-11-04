"""Import aliases for analysis scripts exported by earlier studio versions."""
from importlib import import_module
import sys

from studio_core.api import *
_MODULES = {'agents': 'studio_core.agents', 'analysis_tools': 'studio_core.analysis_tools', 'app': 'studio_core.app', 'cli': 'studio_core.cli', 'data_io': 'studio_core.data_io', 'dml': 'dml', 'example_prompt': 'studio_core.example_prompt', 'exploration': 'exploration', 'forecasting': 'superlearner.forecasting', 'rcf_screen': 'rcf_screen', 'reporting': 'studio_core.reporting', 'sources': 'studio_core.sources', 'spec': 'studio_core.spec', 'studio': 'studio_core.studio', 'web_sources': 'studio_core.web_sources', 'workflow': 'studio_core.workflow'}
for _name, _target in _MODULES.items():
    _module = import_module(_target)
    sys.modules[__name__ + "." + _name] = _module
    globals()[_name] = _module
