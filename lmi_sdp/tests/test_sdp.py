from sympy import Matrix, symbols, pi
from sympy.abc import x, y, z
from numpy import array
from numpy.testing import assert_array_equal

from lmi_sdp import LMI_PSD, LMI_NSD, prepare_lmi_for_sdp, \
    prepare_objective_for_sdp, get_variables, to_cvxopt


def test_prepare_lmi_for_sdp():
    vars = [x, y, z]
    m1 = Matrix([[x, y], [y, z+1]])
    c1 = Matrix([[0, 1], [1, 2]])
    lmi1 = LMI_PSD(m1, c1)
    m2 = Matrix([[y, 0], [0, 2*x]])
    c2 = Matrix([[30, 0], [0, 40]])
    lmi2 = LMI_NSD(m2, c2)
    coeffs = prepare_lmi_for_sdp([lmi1, lmi2], vars,
                                 optimize_by_diag_blocks=True)
    expected = [([array([[1., 0.],
                         [0., 0.]]),
                  array([[0., 1.],
                         [1., 0.]]),
                  array([[0., 0.],
                         [0., 1.]])],
                 array([[0., -1.],
                        [-1., -1.]])),
                ([array([[0.]]),
                  array([[-1.]]),
                  array([[0.]])],
                 array([[30.]])),
                ([array([[-2.]]),
                  array([[0.]]),
                  array([[0.]])],
                 array([[40.]]))]
    for i in range(len(coeffs)):
        assert_array_equal(coeffs[i][0], expected[i][0])
        assert_array_equal(coeffs[i][1], expected[i][1])


def test_prepare_objective_for_sdp():
    vars = [x, y, z]
    assert_array_equal(prepare_objective_for_sdp(1.2 + x - 3.4*y, vars, 'max'),
                       array([-1.0, 3.4, 0.0]))

    except_ok = False
    try:
        prepare_objective_for_sdp(1.2 + x*y, vars)
    except ValueError:
        except_ok = True
    assert except_ok


def test_get_variables():
    x1, x2, x3 = symbols('x1 x2 x3')
    variables = [x1, x2, x3]

    obj = 1.2 + pi*x3
    lmis = [Matrix([x2]), LMI_PSD(Matrix([1.4*x2 + x1]))]

    assert variables == get_variables(obj, lmis)


try:
    import cvxopt
except ImportError:
    pass
else:

    from cvxopt import matrix

    def test_to_cvxopt():
        variables = symbols('x1 x2 x3')
        x1, x2, x3 = variables

        min_obj = x1 - x2 + x3

        LMI_1 = LMI_NSD(
            x1*Matrix([[-7, -11], [-11, 3]]) +
            x2*Matrix([[7, -18], [-18, 8]]) +
            x3*Matrix([[-2, -8], [-8, 1]]),
            Matrix([[33, -9], [-9, 26]]))

        LMI_2 = LMI_NSD(
            x1*Matrix([[-21, -11, 0], [-11, 10, 8], [0, 8, 5]]) +
            x2*Matrix([[0, 10, 16], [10, -10, -10], [16, -10, 3]]) +
            x3*Matrix([[-5, 2, -17], [2, -6, 8], [-17, 8, 6]]),
            Matrix([[14, 9, 40], [9, 91, 10], [40, 10, 15]]))

        ok_c = matrix([1., -1., 1.])
        ok_Gs = [matrix([[-7., -11., -11., 3.],
                        [7., -18., -18., 8.],
                        [-2., -8., -8., 1.]])]
        ok_Gs += [matrix([[-21., -11., 0., -11., 10., 8., 0., 8., 5.],
                          [0., 10., 16., 10., -10., -10., 16., -10., 3.],
                          [-5., 2., -17., 2., -6., 8., -17., 8., 6.]])]
        ok_hs = [matrix([[33., -9.], [-9., 26.]])]
        ok_hs += [matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]])]

        c, Gs, hs = to_cvxopt(min_obj, [LMI_1, LMI_2], variables)

        assert not any(ok_c - c)
        for i in range(len(ok_Gs)):
            assert not any(ok_Gs[i] - Gs[i])
        for i in range(len(ok_hs)):
            assert not any(ok_hs[i] - hs[i])
