"""Tools for symbolic and numerical representations of linear matrices"""
from sympy import ImmutableMatrix, S, diag, Dummy
from sympy.matrices.matrices import MatrixError, NonSquareMatrixError
import numpy as np


class NonLinearMatrixError(ValueError, MatrixError):
    pass


def get_diag_block_idxs(matrix):
    """Get major indexes of diagonal blocks of squared matrices"""
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError('matrix must be square')
    b = []
    n = matrix.shape[0]
    c = 0
    while c < n-1:
        for l in range(n-1, c, -1):
            if matrix[l, c] != 0 or matrix[c, l] != 0:
                break
            elif l == c+1:
                b.append(l)
        c = l
    return b + [n]


def split_by_diag_blocks(matrix):
    """Split a squared matrix into its diagonal blocks"""
    idxs = get_diag_block_idxs(matrix)
    blocks = [matrix[a:b, a:b] for (a, b) in zip([0]+idxs[:-1], idxs)]
    return blocks


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
    dummy = Dummy()

    ok_set = set(variables) | set([S.One, dummy])
    consts = np.zeros((lm.rows, lm.cols))
    coeffs = [np.zeros((lm.rows, lm.cols)) for i in range(len(variables))]
    for elem in [(i, j) for i in range(lm.rows) for j in range(lm.cols)]:
        expr = lm[elem] + dummy  # fixes as_coefficients_dict() behavior for
                                 # single term expressions
        coeff_dict = expr.as_coefficients_dict()
        if not set(coeff_dict.keys()).issubset(ok_set):
            expr = expr.expand()  # try expanding
            coeff_dict = expr.as_coefficients_dict()
            if not set(coeff_dict.keys()).issubset(ok_set):
                raise NonLinearMatrixError(
                    "'linear_matrix' must be composed of linear "
                    "expressions w.r.t. 'variables'")
        consts[elem] = coeff_dict.get(S.One, 0)
        for i, x in enumerate(variables):
            coeffs[i][elem] = coeff_dict.get(x, 0)

    return coeffs, consts


def lm_coeffs_to_sym(coeffs, variables):
    """Create a symbolic matrix linear w.r.t. variables given a list of
    numerical coefficient matrices"""
    lm = ImmutableMatrix(coeffs[1])
    for i, x in enumerate(variables):
        lm += x*ImmutableMatrix(coeffs[0][i])

    return lm
