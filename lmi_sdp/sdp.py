"""Interfaces to SDP solvers"""

from sympy import Basic, Matrix, Dummy, S
from .lm import split_by_diag_blocks, lm_sym_to_coeffs


def prepare_lmi_for_sdp(LMI, variables, optimize_by_diag_blocks=False):

    if isinstance(LMI, Basic) or isinstance(LMI, Matrix):
        LMIs = [LMI]
    else:
        LMIs = list(LMI)

    SLMs = []  # SLM stands for 'Symmetric Linear Matrix'
    for LMI in LMIs:
        if isinstance(LMI, Matrix):
            LM = LMI
        else:
            if isinstance(LMI, Basic) and LMI.is_Relational:
                if LMI.lts == 0:
                    LM = LMI.gts
                elif LMI.gts == 0:
                    LM = -LMI.lts
                else:
                    LM = LMI.gts - LMI.lts
            else:
                raise ValueError('Unknoun LMI data type')
        if not LM.is_symmetric():
            raise ValueError('LMI matrix not symmetric')
        SLMs.append(LM)

    if optimize_by_diag_blocks:
        orig_SLMs = SLMs
        SLMs = []
        for SLM in orig_SLMs:
            SLMs += split_by_diag_blocks(SLM)

    coeffs = [lm_sym_to_coeffs(SLM, variables) for SLM in SLMs]

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
