from sympy import Matrix
import numpy as np
from lmi_sdp import NonSquareMatrixError, get_diag_block_idxs, split_by_diag_blocks


def test_get_diag_block_idxs():
    m = Matrix([[1, 0, 0], [0, 2, 3], [0, 3, 4]])
    assert get_diag_block_idxs(m) == [1, 3]
    m = np.matrix([[1, 0, 0], [0, 2, 3], [0, 3, 4]])
    assert get_diag_block_idxs(m) == [1, 3]
    m = Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert (get_diag_block_idxs(m), get_diag_block_idxs(m.T)) == ([2, 3], [2, 3])
    m = Matrix([[1, 0, 0], [0, 1, 1]])
    try:
        get_diag_block_idxs(m)
        except_ok = False
    except NonSquareMatrixError:
        except_ok = True
    assert except_ok


def test_split_by_diag_blocks():
    m = Matrix([[1, 0, 0], [0, 2, 3], [0, 3, 4]])
    assert split_by_diag_blocks(m) == \
        [Matrix([[1]]), Matrix([[2, 3], [3, 4]])]
    m = np.matrix([[1, 0, 0], [0, 2, 3], [0, 3, 4]])
    mb = split_by_diag_blocks(m)
    assert mb[0] == np.matrix([[1]])
    assert (mb[1] == np.matrix([[2, 3], [3, 4]])).all()
    m = Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert split_by_diag_blocks(m) == \
        [Matrix([[0, 1], [0, 0]]), Matrix([[0]])]
