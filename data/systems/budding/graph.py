"""Extract state graphs from the modular ODE/reset sources without executing them.

Both public graph entry points read the same modular model source.
"""
from pathlib import Path

MODEL_SOURCE_DIR=Path(__file__).resolve().parent


def read_budding_source(source_dir=None):
    root=Path(source_dir or MODEL_SOURCE_DIR).resolve()
    paths=[root/'states.py',root/'parameters.py',root/'ode.py',root/'simulation.py']
    return '\n\n'.join(path.read_text(encoding='utf-8') for path in paths)


def build_budding_graph():
    from ...yeast_graph import build_yeast_graph
    return build_yeast_graph()
