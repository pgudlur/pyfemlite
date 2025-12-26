from pyfemlite.mesh.structured_tri import structured_unit_square_tri

def test_boundary_summary_contains_group_names():
    X, T, boundary = structured_unit_square_tri(3, 2)
    s = boundary.summary()
    assert "Boundary groups:" in s
    for g in ["left", "right", "bottom", "top"]:
        assert g in s
