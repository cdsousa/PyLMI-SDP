"""Interfaces to SDP solvers"""


from sympy import Basic, ordered, sympify, BlockDiagMatrix
from .lm import lin_expr_coeffs, lm_sym_to_coeffs
from .lmi import LMI


class NotAvailableError(Exception):
    def __init__(self, function_name):
        msg = 'Function %s not available since cvxopt package '\
              'was not found' % function_name
        Exception.__init__(self, msg)

try:
    import scipy
except ImportError:  # pragma: no cover
    scipy = None

try:
    import cvxopt
except ImportError:  # pragma: no cover
    cvxopt = None


def lmi_to_coeffs(lmi, variables, split_blocks=False, sparse=False):
    """Transforms LMIs from symbolic to numerical.

    Parameters
    ----------
    lmi: symbolic LMI or Matrix, or a list of them
    variables: list of symbols
    split_blocks: bool or string
        If set to True, function tries to subdivide each LMI into
        smaller diagonal blocks. If set to 'BlockDiagMatrix',
        BlockDiagMatrix's are split into their diagonal blocks but the
        funtion does not try to subdivide them any further.
    sparse: bool
        Set whether return matrices dense or sparse. Dense by default.

    Returns
    -------
    coeffs: list of numerical LMIs
        List of numerical LMIs where each one is a pair where the first
        element is a list of numpy arrays corresponding to the coefficients of
        each variable, and the second element is a numpy array with zero order
        coefficients (constants not  multipling by any variable). The
        numerical coefficients are extracted from the matrix `M` of the
        canonical PSD (or PD) LMI form `M>=0` (or `M>0`).

    Example
    -------
    >>> from sympy import Matrix
    >>> from sympy.abc import x, y, z
    >>> from lmi_sdp import LMI_PSD, lmi_to_coeffs
    >>> vars = [x, y, z]
    >>> m = Matrix([[x+3, y-2], [y-2, z]])
    >>> lmi = LMI_PSD(m)
    >>> lmi_to_coeffs(lmi, vars)
    [([array([[ 1.,  0.],
           [ 0.,  0.]]), array([[ 0.,  1.],
           [ 1.,  0.]]), array([[ 0.,  0.],
           [ 0.,  1.]])], array([[ 3., -2.],
           [-2.,  0.]]))]
    """
    if isinstance(lmi, Basic):
        lmis = [lmi]
    else:
        lmis = list(lmi)

    slms = []  # SLM stands for 'Symmetric Linear Matrix'
    for lmi in lmis:
        if lmi.is_Matrix:
            lmi = LMI(lmi)
        lm = lmi.canonical().gts
        slms.append(lm)

    if split_blocks:
        orig_slms = slms
        slms = []
        for slm in orig_slms:
            if isinstance(slm, BlockDiagMatrix):
                if split_blocks == 'BlockDiagMatrix':
                    slms += slm.diag
                else:
                    slms += sum([d.get_diag_blocks() for d in slm.diag], [])
            else:
                slms += slm.get_diag_blocks()

    coeffs = [lm_sym_to_coeffs(slm, variables, sparse) for slm in slms]

    return coeffs


def objective_to_coeffs(objective_func, variables,
                        objective_type='minimize'):
    """Extracts variable coefficients from symbolic minimization objective
    funtion.

    Parameters
    ----------
    objective_func: symbolic linear expression
    variables: list of symbols
    objective_type: 'maximize' or 'minimize', defaults to 'minimize'

    Returns
    -------
    coeffs: list
        List of coefficients which multiply by the variables of them
        *minimization* function. If the input is a maximization function
        then the output coefficients will be symmetric to the expression
        ones.

    Example
    -------
    >>> from sympy.abc import x, y, z
    >>> from lmi_sdp import objective_to_coeffs
    >>> vars = [x, y, z]
    >>> expr = 1.1 + x + 2.2*y
    >>> objective_to_coeffs(expr, vars)
    [1.0, 2.2, 0.0]
    >>> objective_to_coeffs(expr, vars, 'maximize')
    [-1.0, -2.2, 0.0]
    """
    objective_type = objective_type.lower()
    if objective_type in ['max', 'maximize']:
        objective_func = -1 * objective_func
    elif objective_type in ['min', 'minimize']:
        pass
    else:
        raise ValueError("objective_type must be 'maximize' or 'minimize'")

    coeffs, const = lin_expr_coeffs(objective_func, variables)

    return coeffs


def get_variables(objective_func=0, lmis=None):
    """Extract free variables from objective_func and lmis.
    """
    if lmis is None:
        lmis = []
    variables = sympify(objective_func).free_symbols
    for lmi in lmis:
        if lmi.is_Matrix:
            lm = lmi
        else:
            lm = lmi.canonical().gts
        for expr in lm:
            variables |= expr.free_symbols
    return list(ordered(variables))


def to_cvxopt(objective_func, lmis, variables, objective_type='minimize',
              split_blocks=True):
    """Prepare objective and LMI to be used with cvxopt SDP solver.

    Parameters
    ----------
    objective_func: symbolic linear expression
    lmi: symbolic LMI or Matrix, or a list of them
    variables: list of symbols
        The variable symbols which form the LMI/SDP space.
    objective_type: 'maximize' or 'minimize', defaults to 'minimize'
    split_blocks: bool
        If set to True, function tries to subdivide each LMI into
        smaller diagonal blocks

    Returns
    -------
    c, Gs, hs: parameters ready to be input to cvxopt.solvers.sdp()
    """
    if cvxopt is None:
        raise NotAvailableError(to_cvxopt.__name__)

    obj_coeffs = objective_to_coeffs(objective_func, variables,
                                     objective_type)
    lmi_coeffs = lmi_to_coeffs(lmis, variables, split_blocks, sparse=False)

    c = cvxopt.matrix(obj_coeffs)

    Gs = []
    hs = []

    for (LMis, LM0) in lmi_coeffs:
        #Gs.append([-LMi for LMi in LMis])
        #hs.append(LM0)
        Gs.append(cvxopt.matrix([(-LMi).flatten().astype(float).tolist()
                                 for LMi in LMis]))
        hs.append(cvxopt.matrix(LM0.astype(float).tolist()))

    return c, Gs, hs


def _sdpa_header(obj_coeffs, lmi_coeffs, comment=None):
    """Helper funtion to generate headers of SDPA files."""
    s = '"' + comment + '"\n' if comment is not None else ''
    s += str(len(obj_coeffs)) + ' = ndim\n'
    s += str(len(lmi_coeffs)) + ' = nblocks\n'
    for block in lmi_coeffs:
        s += str(block[0][1].shape[0]) + ' '
    s += '= blockstruct\n'
    for x in obj_coeffs:
        s += str(x) + ', '
    s = s[:-2] + ' = objcoeffs\n'
    return s


def to_sdpa_sparse(objective_func, lmis, variables, objective_type='minimize',
                   split_blocks=True, comment=None):
    """Put problem (objective and LMIs) into SDPA sparse format."""
    obj_coeffs = objective_to_coeffs(objective_func, variables,
                                     objective_type)
    lmi_coeffs = lmi_to_coeffs(lmis, variables, split_blocks,
                               sparse=(True if scipy else False))

    s = _sdpa_header(obj_coeffs, lmi_coeffs, comment)

    if scipy:
        def _print_sparse(x, b, m, sign=1):
            s = ''
            nzi, nzj = m.nonzero()
            for idx, i in enumerate(nzi):
                j = nzj[idx]
                if j >= i:
                    e = sign*m[i, j]
                    s += '%d %d %d %d %s\n' % (x, b, i+1, j+1, str(e))
            return s
    else:
        def _print_sparse(x, b, m, sign=1):
            s = ''
            shape = m.shape
            for i in range(shape[0]):
                for j in range(i, shape[1]):
                    e = sign*m[i, j]
                    if e != 0:
                        s += '%d %d %d %d %s\n' % (x, b, i+1, j+1, str(e))
            return s

    for b in range(len(lmi_coeffs)):
        s += _print_sparse(0, b+1, lmi_coeffs[b][1], sign=-1)
    for x in range(len(obj_coeffs)):
        for b in range(len(lmi_coeffs)):
            s += _print_sparse(x+1, b+1, lmi_coeffs[b][0][x])

    return s


def to_sdpa_dense(objective_func, lmis, variables, objective_type='minimize',
                  split_blocks=True, comment=None):
    """Put SDP problem (objective and LMIs) into SDPA dense format."""
    obj_coeffs = objective_to_coeffs(objective_func, variables,
                                     objective_type)
    lmi_coeffs = lmi_to_coeffs(lmis, variables, split_blocks, sparse=False)

    s = _sdpa_header(obj_coeffs, lmi_coeffs, comment)

    def _print_dense(m, sign=1):
        s = '\n {'
        shape = m.shape
        for i in range(shape[0]):
            s += '\n  {'
            for j in range(shape[1]):
                s += ' ' + str(sign*m[i, j]) + ','
            s = s[:-1] + ' },'
        s = s[:-1] + '\n }'
        return s

    s += '{'
    for b in range(len(lmi_coeffs)):
        s += _print_dense(lmi_coeffs[b][1], sign=-1)
    s += '\n}\n'
    for x in range(len(obj_coeffs)):
        s += '{'
        for b in range(len(lmi_coeffs)):
            s += _print_dense(lmi_coeffs[b][0][x])
        s += '\n}\n'

    return s
