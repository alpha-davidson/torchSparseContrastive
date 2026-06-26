"""
sparse_simclr.py
----------------
Thin shim that re-exports SparseSimCLR and sparse_simclr_21d from model.py
so that train_contrastive.py can do:

    from sparse_simclr import sparse_simclr_21d, SparseSimCLR
"""

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("local_model", str(_HERE / "model.py"))
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["local_model"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]

SparseSimCLR     = _mod.SparseSimCLR
sparse_simclr_21d = _mod.sparse_simclr_21d

__all__ = ["SparseSimCLR", "sparse_simclr_21d"]
