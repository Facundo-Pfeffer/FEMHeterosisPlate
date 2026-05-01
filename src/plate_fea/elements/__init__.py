"""
Plate element formulations.

Currently exports HeterosisPlateElement, the Q8/Q9 selective-integration quadrilateral
from Hughes & Cohen (1978). The abstract base class PlateElementBase (elements/base.py)
defines the interface that any element must implement to work with the assembly and solver.
"""

from .heterosis_plate import HeterosisPlateElement

__all__ = ["HeterosisPlateElement"]
