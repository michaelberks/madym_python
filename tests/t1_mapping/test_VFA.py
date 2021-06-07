import pytest
import os
import sys
import numpy as np

sys.path.insert(0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from QbiPy.t1_mapping import VFA

@pytest.mark.parametrize("T1, S0, FA, TR, output_shape",
    [
        (1000, 2000, [2,10, 18], 4, (1,3)),
        ([1000,1010], [2000,2000], [2,10, 18], 4, (2,3)),
        ([1000,1010], [2000,2000], [[2,10, 18],[2,8,16]], 4, (2,3)),
        (1000, [2000,2000], [[2,10, 18],[2,8,16]], 4, (2,3)),
        (1000, [2000,2000], [[2,10, 18],[2,8,16]], [4,4.5], (2,3)),
    ])
def test_signal_from_T1(T1, S0, FA, TR, output_shape):
    
    signal = VFA.signal_from_T1(T1, S0, FA, TR)
    assert signal.shape == output_shape

