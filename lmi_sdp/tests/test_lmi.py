from sympy import Matrix, S
from sympy.abc import x, y, z
from lmi_sdp.lmi import LMI_PSD, ShapeError, NonSymmetricMatrixError


def test_LMI_PSD():
    m = Matrix([[x, y], [y, z+1]])
    lmi = LMI_PSD(m)
    assert lmi.lhs == lmi.gts == m
    assert lmi.rhs == lmi.lts == S(0)

    c = Matrix([[0, 1], [1, 2]])
    lmi = LMI_PSD(m, c)
    assert lmi.lhs == lmi.gts == m
    assert lmi.rhs == lmi.lts == c


def test_LMI_PSD_exceptions():
    except_ok = False
    try:
        LMI_PSD(Matrix([[1, x], [y, z]]))
    except NonSymmetricMatrixError:
        except_ok = True
    assert except_ok

    except_ok = False
    try:
        LMI_PSD(Matrix([[x+y]]), Matrix([[x, y], [y, z+1]]))
    except ShapeError:
        except_ok = True
    assert except_ok
