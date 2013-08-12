"""Tools for symbolic and numerical representations of linear matrices"""
from sympy import Matrix, S, diag, Dummy
import numpy as np


class NonSquareMatrixError(ValueError):
    pass


class NonLinearMatrixError(ValueError):
    pass


def get_diag_block_idxs(M):
    """Get major indexes of diagonal blocks of squared matrices"""
    if M.shape[0] != M.shape[1]:
        raise NonSquareMatrixError('matrix must be square')
    b = []
    n = M.shape[0]
    c = 0
    while c < n-1:
        for l in range(n-1, c, -1):
            if M[l, c] != 0 or M[c, l] != 0:
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
    numerical coefficient matrices"""
    LM = linear_matrix
    X = variables
    nx = len(X)
    dummy = Dummy()

    one = S(1)
    ok_set = set(X) | set([one, dummy])
    consts = np.zeros((LM.rows, LM.cols))
    coeffs = [np.zeros((LM.rows, LM.cols)) for i in range(nx)]
    for elem in [(i, j) for i in range(LM.rows) for j in range(LM.cols)]:
        expr = LM[elem] + dummy  # fixes as_coefficients_dict() behavior for
                                 # single term expressions
        coeff_dict = expr.as_coefficients_dict()
        if not set(coeff_dict.keys()).issubset(ok_set):
            expr = expr.expand()  # try expanding
            coeff_dict = expr.as_coefficients_dict()
            if not set(coeff_dict.keys()).issubset(ok_set):
                raise NonLinearMatrixError(
                    "'linear_matrix' must be composed of linear "
                    "expressions w.r.t. 'variables'")
        consts[elem] = coeff_dict.get(one, 0)
        for i, x in enumerate(X):
            coeffs[i][elem] = coeff_dict.get(x, 0)

    return (coeffs, consts)


def lm_coeffs_to_sym(coeffs, variables):
    """Create a symbolic matrix linear w.r.t. variables given a list of
    numerical coefficient matrices"""
    X = variables
    nx = len(X)

    LM = Matrix(coeffs[1])
    for i, x in enumerate(X):
        LM += x*Matrix(coeffs[0][i])

    return LM
