from sympy import Matrix, factor, zeros
from sympy.abc import x, y, z
import numpy as np
from lmi_sdp import NonSquareMatrixError, NonLinearMatrixError, \
    get_diag_block_idxs, split_by_diag_blocks, lm_sym_to_coeffs, \
    lm_coeffs_to_sym


def test_get_diag_block_idxs():
    m = Matrix([[1, 0, 0], [0, 2, 3], [0, 3, 4]])
    assert get_diag_block_idxs(m) == [1, 3]

    m = np.matrix([[1, 0, 0], [0, 2, 3], [0, 3, 4]])
    assert get_diag_block_idxs(m) == [1, 3]

    m = Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert (get_diag_block_idxs(m), get_diag_block_idxs(m.T)) == \
        ([2, 3], [2, 3])


def test_get_diag_block_idxs_exceptions():
    m = Matrix([[1, 0, 0], [0, 1, 1]])
    except_ok = False
    try:
        get_diag_block_idxs(m)
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


def test_lm_sym_to_coeffs():
    m = Matrix([[1.2, x], [3.4*y, 1.2 + 3*x - 4.5*y + z]])
    coeffs = lm_sym_to_coeffs(m, [x, y, z])
    assert len(coeffs) == 2
    assert len(coeffs[0]) == 3
    assert (coeffs[0][0] == np.matrix([[0.0, 1.0], [0.0, 3.0]])).all()
    assert (coeffs[0][1] == np.matrix([[0.0, 0.0], [3.4, -4.5]])).all()
    assert (coeffs[0][2] == np.matrix([[0.0, 0.0], [0.0, 1.0]])).all()
    assert (coeffs[1] == np.matrix([[1.2, 0.0], [0.0, 1.2]])).all()

    assert lm_sym_to_coeffs(Matrix([0.0]), [x, y, z]) == \
        ([np.matrix([[0.0]]), np.matrix([[0.0]]), np.matrix([[0.0]])],
         np.matrix([[0.0]]))


def test_lm_sym_to_coeffs_exceptions():
    except_ok = False
    try:
        lm_sym_to_coeffs(Matrix([1.2 + x + y*z]), [x, y, z])
    except NonLinearMatrixError:
        except_ok = True
    assert except_ok

    except_ok = False
    try:
        lm_sym_to_coeffs(Matrix([1.2 + x*y]), [x])
    except NonLinearMatrixError:
        except_ok = True
    assert except_ok


def test_lm_coeffs_to_sym():
    var_coeffs = [None]*3
    var_coeffs[0] = np.matrix([[0.0, 1.0], [0.0, 3.0]])
    var_coeffs[1] = np.matrix([[0.0, 0.0], [3.4, -4.5]])
    var_coeffs[2] = np.matrix([[0.0, 0.0], [0.0, 1.0]])
    consts = np.matrix([[1.2, 0.0], [0.0, 1.2]])
    coeffs = (var_coeffs, consts)
    m = Matrix([[1.2, x], [3.4*y, 1.2 + 3*x - 4.5*y + z]])
    assert lm_coeffs_to_sym(coeffs, [x, y, z]) - m == zeros(2)
