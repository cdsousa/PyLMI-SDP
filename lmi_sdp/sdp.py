"""Interfaces to SDP solvers"""

from sympy import Basic, Matrix, Dummy, S
from .lm import split_by_diag_blocks, lm_sym_to_coeffs
from .lmi import _LMI


def prepare_lmi_for_sdp(lmi, variables, optimize_by_diag_blocks=False):

    if isinstance(lmi, _LMI) or isinstance(lmi, Matrix):
        lmis = [lmi]
    else:
        lmis = list(lmi)

    slms = []  # SLM stands for 'Symmetric Linear Matrix'
    for lmi in lmis:
        if isinstance(lmi, Matrix):
            if not lmi.is_symmetric():
                raise ValueError('LMI matrix not symmetric')
            lm = lmi
        else:
            if isinstance(lmi, _LMI):
                if lmi.lts == 0:
                    lm = lmi.gts
                elif lmi.gts == 0:
                    lm = -lmi.lts
                else:
                    lm = lmi.gts - lmi.lts
            else:
                raise ValueError('Unknoun LMI data type')
        slms.append(lm)

    if optimize_by_diag_blocks:
        orig_slms = slms
        slms = []
        for slm in orig_slms:
            slms += split_by_diag_blocks(slm)

    coeffs = [lm_sym_to_coeffs(slm, variables) for slm in slms]

    return coeffs


def prepare_objective_for_sdp(objective_type, objective_func, variables):
    objective_type = objective_type.lower()
    if objective_type in ['max', 'maximize']:
        objective_func = -1 * objective_func
    elif objective_type in ['min', 'minimize']:
        pass
    else:
        raise ValueError("objective_type must be 'maximize' or 'minimize'")

    dummy = Dummy()
    objective_func += dummy  # fixes as_coefficients_dict() behavior for
                             # single term expressions

    coeff_dict = objective_func.as_coefficients_dict()

    ok_set = set(variables) | set([S(1), dummy])
    if not set(coeff_dict.keys()).issubset(ok_set):
        raise ValueError("objective_func must be linear w.r.t. variables")

    coeffs = [coeff_dict.get(x, 0) for x in variables]

    return coeffs
