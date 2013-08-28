from sympy import Matrix, S, latex
from sympy.abc import x, y, z
from lmi_sdp.lmi import ShapeError, NonSymmetricMatrixError, LMI_PSD, \
    LMI_NSD, LMI_PD, LMI_ND, init_lmi_latex_printing


def test_LMI_PSD():
    m = Matrix([[x, y], [y, z+1]])
    lmi = LMI_PSD(m)
    assert lmi.lhs == lmi.gts == m
    assert lmi.rhs == lmi.lts == S(0)

    c = Matrix([[0, 1], [1, 2]])
    lmi = LMI_PSD(m, c)
    assert lmi.lhs == lmi.gts == m
    assert lmi.rhs == lmi.lts == c


def test_LMI_canonical():
    m = Matrix([[x, y], [y, z+1]])
    c = Matrix([[0, 1], [1, 2]])
    can_lhs = m-c
    can_rhs = 0

    can = LMI_PSD(m, c).canonical()
    assert can.lhs == can.gts == can_lhs
    assert can.rhs == can.lts == can_rhs

    can = LMI_PSD(0, c-m).canonical()
    assert can.lhs == can.gts == can_lhs
    assert can.rhs == can.lts == can_rhs

    can = LMI_PD(m, c).canonical()
    assert can.lhs == can.gts == can_lhs
    assert can.rhs == can.lts == can_rhs

    can = LMI_NSD(c, m).canonical()
    assert can.lhs == can.gts == can_lhs
    assert can.rhs == can.lts == can_rhs
    assert isinstance(can, LMI_PSD)

    can = LMI_ND(c, m).canonical()
    assert can.lhs == can.gts == can_lhs
    assert can.rhs == can.lts == can_rhs
    assert isinstance(can, LMI_PD)


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


def test_lmi_latex_printing():

    init_lmi_latex_printing()

    lmi = LMI_PSD(Matrix([[x, y], [y, z+1]]))

    assert r"\succeq 0" == latex(lmi)[-9:]
