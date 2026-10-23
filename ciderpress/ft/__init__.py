"""Frozen CIDER mean plus a residual GP on fixed-density energy records.

``ReactionSet`` handles energy units, stoichiometry and splits. ``FineTuner``
wraps MOLGP2/DFTKernel2 and saves ordinary additive MappedXC2 YAML models.
See README.md for record conventions, constraints and the SCF callback contract.
"""

from .data import ReactionSet
from .records import from_pyscf
from .tuner import Constraints, FineTuner

__all__ = ["Constraints", "FineTuner", "ReactionSet", "from_pyscf"]
