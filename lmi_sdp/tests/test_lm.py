from sympy import Matrix
import numpy as np
from lmi_sdp import get_diag_blocks, NonSquareMatrixError


def test_get_diag_blocks():
    m = Matrix([[1, 0, 0], [0, 2, 3], [0, 3, 4]])
    assert get_diag_blocks(m) == [1, 3]
    m = np.matrix([[1, 0, 0], [0, 2, 3], [0, 3, 4]])
    assert get_diag_blocks(m) == [1, 3]
    m = Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert (get_diag_blocks(m), get_diag_blocks(m.T)) == ([2, 3], [2, 3])
    m = Matrix([[1, 0, 0], [0, 1, 1]])
    try:
        get_diag_blocks(m)
        except_ok = False
    except NonSquareMatrixError:
        except_ok = True
    assert except_ok
