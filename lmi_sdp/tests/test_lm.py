from sympy import Matrix, factor, zeros, MatAdd, MatMul
from sympy.abc import x, y, z
import numpy as np
from lmi_sdp import NonLinearExpressionError, NonSquareMatrixError, NonLinearMatrixError, \
    lin_expr_coeffs, lm_sym_to_coeffs, lm_coeffs_to_sym, lm_sym_expanded


def test_lin_expr_coeffs():
    e = 1.2 + 3*x - 4.5*y + z
    coeffs, const = lin_expr_coeffs(e, [x, y, z])
    assert coeffs == [3.0, -4.5, 1.0]
    assert const == 1.2


def test_lin_expr_coeffs_exceptions():
    except_ok = False
    try:
        lin_expr_coeffs(1.2 + x + y*z, [x, y, z])
    except NonLinearExpressionError:
        except_ok = True
    assert except_ok

    except_ok = False
    try:
        lin_expr_coeffs(1.2 + x*y, [x])
    except NonLinearExpressionError:
        except_ok = True
    assert except_ok


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


def test_lm_sym_expanded():
    m = Matrix([[0, x], [3.4*y, 3*x - 4.5*y + z]])
    c = Matrix([[1.2, 0], [0, 1.2]])
    cx = MatMul(Matrix([[0.0, 1.0], [0.0, 3.0]]), x)
    cy = MatMul(Matrix([[0.0, 0.0], [3.4, -4.5]]), y)
    cz = MatMul(Matrix([[0.0, 0.0], [0.0, 1.0]]), z)
    cc = Matrix([[1.2, 0.0], [0.0, 1.2]])
    assert MatAdd(cx, cy, cz, cc) == lm_sym_expanded(m+c, [x, y, z])
    assert MatAdd(cx, cy, cz) == lm_sym_expanded(m, [x, y, z])
    assert cc == lm_sym_expanded(c, [x, y, z])
