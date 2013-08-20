"""Interfaces to SDP solvers"""

from sympy import Basic, Matrix, Dummy, S
from numpy import array
from .lm import lin_expr_coeffs, split_by_diag_blocks, lm_sym_to_coeffs
from .lmi import BaseLMI, LMI


def prepare_lmi_for_sdp(lmi, variables, optimize_by_diag_blocks=False):
    """Transforms LMIs from symbolic to numerical.

    Parameters
    ----------
    lmi: symbolic LMI or Matrix, or a list of them
    variables: list of symbols
    optimize_by_diag_blocks: bool
        If set to True, function tries to subdivide each LMI into
        smaller diagonal blocks

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
    >>> from lmi_sdp import LMI_PSD, prepare_lmi_for_sdp
    >>> vars = [x, y, z]
    >>> m = Matrix([[x+3, y-2], [y-2, z]])
    >>> lmi = LMI_PSD(m)
    >>> prepare_lmi_for_sdp(lmi, vars)
    [([array([[ 1.,  0.],
           [ 0.,  0.]]), array([[ 0.,  1.],
           [ 1.,  0.]]), array([[ 0.,  0.],
           [ 0.,  1.]])], array([[ 3., -2.],
           [-2.,  0.]]))]
    """
    if isinstance(lmi, BaseLMI) or isinstance(lmi, Matrix):
        lmis = [lmi]
    else:
        lmis = list(lmi)

    slms = []  # SLM stands for 'Symmetric Linear Matrix'
    for lmi in lmis:
        if isinstance(lmi, Matrix):
            lmi = LMI(lmi)
        lm = lmi.canonical().gts
        slms.append(lm)

    if optimize_by_diag_blocks:
        orig_slms = slms
        slms = []
        for slm in orig_slms:
            slms += split_by_diag_blocks(slm)

    coeffs = [lm_sym_to_coeffs(slm, variables) for slm in slms]

    return coeffs


def prepare_objective_for_sdp(objective_func, variables,
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
    coeffs: numpy array
        List of coefficients which multiply by the variables of them
        *minimization* function. If the input is a maximization function
        then the output coefficients will be symmetric to the expression
        ones.

    Example
    -------
    >>> from sympy.abc import x, y, z
    >>> from lmi_sdp import prepare_objective_for_sdp
    >>> vars = [x, y, z]
    >>> expr = 1.1 + x + 2*y
    >>> prepare_objective_for_sdp(expr, vars)
    array([ 1.,  2.,  0.])
    >>> prepare_objective_for_sdp(expr, vars, 'maximize')
    array([-1., -2.,  0.])
    """
    objective_type = objective_type.lower()
    if objective_type in ['max', 'maximize']:
        objective_func = -1 * objective_func
    elif objective_type in ['min', 'minimize']:
        pass
    else:
        raise ValueError("objective_type must be 'maximize' or 'minimize'")

    coeffs, const = lin_expr_coeffs(objective_func, variables)

    return array(coeffs).astype(float)
