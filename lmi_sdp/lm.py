"""Tools for symbolic and numerical representations of linear matrices"""
from sympy import Matrix, S, diag
import numpy as np


class NonSquareMatrixError(ValueError):
    pass


class NonLinearMatrixError(ValueError):
    pass


def get_diag_blocks(M):
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


def lm_sym_to_coeffs(linear_matrix, variables, split_diag_blocks=False):
    """Convert a symbolic matrix linear w.r.t. variables into a list of
    numerical coefficient matrices"""
    LM = linear_matrix
    X = variables
    nx = len(X)

    if isinstance(LM, list):
        LM_blocks = LM
    elif isinstance(LM, Matrix):
        if split_diag_blocks:
            blocks = get_diag_blocks(LM)
            LM_blocks = []
            for a, b in zip([0]+blocks[:-1], blocks):
                LM_blocks.append(LM[a:b, a:b])
        else:
            LM_blocks = [LM]

    coeffs_blocks = []

    one = S(1)
    ok_set = set(X) | set([one])
    for LM_b in LM_blocks:
        consts = np.zeros((LM_b.rows, LM_b.cols))
        coeffs = [np.zeros((LM_b.rows, LM_b.cols)) for i in range(nx)]
        for elem in [(i, j) for i in range(LM_b.rows)
                     for j in range(LM_b.cols)]:
            expr = LM_b[elem]
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
        coeffs_blocks.append((coeffs, consts))

    if isinstance(LM, list) or split_diag_blocks:
        return coeffs_blocks
    else:
        return coeffs_blocks[0]


def lm_coeffs_to_sym(coeffs, variables, join_blocks=False):
    """Create a symbolic matrix linear w.r.t. variables given a list of
    numerical coefficient matrices"""
    X = variables
    nx = len(X)

    if isinstance(coeffs, list):
        coeffs_blocks = coeffs
    else:
        coeffs_blocks = [coeffs]

    LM_blocks = []

    for _coeffs in coeffs_blocks:
        LM_block = Matrix(_coeffs[1])
        for i, x in enumerate(X):
            LM_block += x*Matrix(_coeffs[0][i])
        LM_blocks.append(LM_block)

    if isinstance(coeffs, list):
        if join_blocks:
            return diag(*LM_blocks)
        else:
            return LM_blocks
    else:
        return LM_blocks[0]
