# slm/brains/__init__.py

from .cfo_slm import run as run_cfo
from .cmo_slm import run as run_cmo
from .coo_slm import run as run_coo
from .chro_slm import run as run_chro
from .cpo_slm import run as run_cpo
from . import ea_slm  # keep module import for EA coordinator

__all__ = [
    "run_cfo", "run_cmo", "run_coo", "run_chro", "run_cpo", "ea_slm"
]
