"""Tools for symbolic and numerical representations of linear matrices"""
from sympy import ImmutableMatrix, S, diag, Dummy, MatMul, MatAdd, S
from sympy.matrices.matrices import MatrixError, NonSquareMatrixError
import numpy as np


class NonLinearExpressionError(ValueError):
    pass


class NonLinearMatrixError(ValueError, MatrixError):
    pass


def lin_expr_coeffs(linear_expr, variables):
    """Convert a symbolic expression linear w.r.t. variables into a list of
    numerical coefficient

    Returns
    -------

    coeffs: list of floats
        List of coefficients of each variable in variables.
    consts: float
        The constant term (zero order coefficient).
    """
    dummy = Dummy()
    ok_set = set(variables) | set([S.One, dummy])
    expr = dummy + linear_expr  # fixes as_coefficients_dict() behavior for
                                # single term expressions
    coeff_dict = expr.as_coefficients_dict()
    if not set(coeff_dict.keys()).issubset(ok_set):
        expr = expr.expand()  # try expanding
        coeff_dict = expr.as_coefficients_dict()
        if not set(coeff_dict.keys()).issubset(ok_set):
                raise NonLinearExpressionError(
                    "'linear_expr' must be linear w.r.t. 'variables'")
    const = float(coeff_dict.get(S.One, 0))
    coeffs = [float(coeff_dict.get(x, 0)) for x in variables]
    return coeffs, const


def lm_sym_to_coeffs(linear_matrix, variables):
    """Convert a symbolic matrix linear w.r.t. variables into a list of
    numerical coefficient matrices

    Returns
    -------

    coeffs: list of numpy matrices
        List of numpy matrices, each containing the coefficients of each
        variable in variables.
    consts: numpy matrix
        Matrix containing the constant terms (zero order coefficients).
    """
    lm = linear_matrix

    consts = np.zeros((lm.rows, lm.cols))
    coeffs = [np.zeros((lm.rows, lm.cols)) for i in range(len(variables))]
    for elem in [(i, j) for i in range(lm.rows) for j in range(lm.cols)]:
        if lm[elem] != 0:
            try:
                coeffs_elem, consts[elem] = lin_expr_coeffs(lm[elem], variables)
            except NonLinearExpressionError:
                raise NonLinearMatrixError(
                    "'linear_matrix' must be composed of linear "
                    "expressions w.r.t. 'variables'")
            for i in range(len(variables)):
                coeffs[i][elem] = coeffs_elem[i]
    return coeffs, consts


def lm_coeffs_to_sym(coeffs, variables):
    """Create a symbolic matrix linear w.r.t. variables given a list of
    numerical coefficient matrices"""
    lm = ImmutableMatrix(coeffs[1])
    for i, x in enumerate(variables):
        lm += x*ImmutableMatrix(coeffs[0][i])

    return lm


def lm_sym_expanded(linear_matrix, variables):
    """Return matrix in the form of sum of coefficent matrices times varibles.
    """
    if S(linear_matrix).free_symbols & set(variables):
        coeffs, const = lm_sym_to_coeffs(linear_matrix, variables)
        terms = []
        for i, v in enumerate(variables):
            terms.append(MatMul(ImmutableMatrix(coeffs[i]), v))
        if const.any():
            terms.append(ImmutableMatrix(const))
        return MatAdd(*terms)
    else:
        return linear_matrix
