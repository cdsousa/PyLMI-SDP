"""Tools for symbolic and numerical representations of linear matrices"""
from sympy import ImmutableMatrix, S, Dummy, MatMul, MatAdd
from sympy.matrices.matrices import MatrixError
from numpy import zeros

try:
    import scipy
except ImportError:
    scipy = None
else:
    import scipy.sparse

try:
    import cvxopt
except ImportError:
    cvxopt = None


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


def lm_sym_to_coeffs(linear_matrix, variables, sparse=False):
    """Convert a symbolic matrix linear w.r.t. variables into a list of
    numerical coefficient matrices

    Parameters
    ----------
    linear_matrix: symbolic linear matrix
    variables: list of symbols
    sparse: bool or string
        Set whether return matrices are sparse or dense. If set to False,
        (the default) numpy.matrix dense matrices are used. If set to True,
        scipy.sparse.lil_matrix sparse matrices are used. If set to 'cvxopt',
        cvxopt.sparse.spmatrix sparse matrices are used.

    Returns
    -------

    coeffs: list of numpy matrices
        List of numpy matrices, each containing the coefficients of each
        variable in variables.
    consts: numpy matrix
        Matrix containing the constant terms (zero order coefficients).
    """

    lm = linear_matrix

    if scipy and sparse is True:
        consts = scipy.sparse.lil_matrix((lm.rows, lm.cols))
        coeffs = [scipy.sparse.lil_matrix((lm.rows, lm.cols))
                  for i in range(len(variables))]
    elif cvxopt and sparse == 'cvxopt':
        consts = cvxopt.spmatrix([], [], [], (lm.rows, lm.cols))
        coeffs = [cvxopt.spmatrix([], [], [], (lm.rows, lm.cols))
                  for i in range(len(variables))]
    else:
        consts = zeros((lm.rows, lm.cols))
        coeffs = [zeros((lm.rows, lm.cols)) for i in range(len(variables))]

    for elem in [(i, j) for i in range(lm.rows) for j in range(lm.cols)]:
        if lm[elem] != 0:
            try:
                coeffs_elem, consts[elem] = lin_expr_coeffs(lm[elem],
                                                            variables)
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
