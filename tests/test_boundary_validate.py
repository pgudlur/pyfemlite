import numpy as np
import pytest
from pyfemlite.mesh.structured_tri import structured_unit_square_tri
from pyfemlite.mesh.boundary import Boundary

def test_boundary_validate_passes_structured():
    X, T, boundary = structured_unit_square_tri(10, 6)
    boundary.validate(X, T, strict=True)

def test_boundary_validate_catches_out_of_range_node():
    X, T, boundary = structured_unit_square_tri(4, 3)
    bad_nodes = boundary.nodes["left"].copy()
    bad_nodes[0] = X.shape[0] + 10
    bad = Boundary(nodes={**boundary.nodes, "left": bad_nodes}, edges=boundary.edges)
    with pytest.raises(ValueError):
        bad.validate(X, T, strict=True)
