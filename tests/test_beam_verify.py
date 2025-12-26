import numpy as np
from pyfemlite.post.beam_verify import euler_bernoulli_tip_deflection_end_traction

def test_euler_bernoulli_formula():
    T0, L, H, E = 2.0, 3.0, 1.5, 100.0
    delta = euler_bernoulli_tip_deflection_end_traction(T0=T0, L=L, H=H, E=E)
    expected = 4.0 * T0 * (L**3) / (E * (H**2))
    assert abs(delta - expected) < 1e-15
