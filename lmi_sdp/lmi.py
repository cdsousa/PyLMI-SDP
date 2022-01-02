"""LMI representation and tools"""

import sympy
from sympy import sympify, GreaterThan, StrictGreaterThan, LessThan, \
    StrictLessThan, MatrixExpr, block_collapse

from sympy.matrices.matrices import MatrixError, ShapeError

from .lm import lm_sym_expanded

from packaging import version


class NonSymmetricMatrixError(ValueError, MatrixError):
    pass


class BaseLMI(object):
    """Not intended for general use

    BaseLMI is used so that LMI_* classes can share functions.
    """
    _options = dict(evaluate=False) if version.parse(sympy.__version__) >= version.parse("0.7.6") else dict()

    def __new__(cls, lhs, rhs, rel_cls, assert_symmetry=True):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        if assert_symmetry:
            if lhs.is_Matrix and hasattr(lhs, 'is_symmetric') and \
                    not lhs.is_symmetric():
                raise NonSymmetricMatrixError('lhs matrix is not symmetric')
            if rhs.is_Matrix and hasattr(rhs, 'is_symmetric') and \
                    not rhs.is_symmetric():
                raise NonSymmetricMatrixError('rsh matrix is not symmetric')
        if lhs.is_Matrix and rhs.is_Matrix:
                if lhs.shape != rhs.shape:
                    raise ShapeError('LMI matrices have different shapes')
        elif not ((lhs.is_Matrix and rhs.is_zero)
                  or (lhs.is_zero and rhs.is_Matrix)):
            raise ValueError('LMI sides must be two matrices '
                             'or a matrix and a zero')

        return rel_cls.__new__(cls, lhs, rhs, **BaseLMI._options)

    def canonical(self):
        """Returns the LMI positive (semi-)definite form with the matrix at
        the rhs and zero at the lhs.
        """
        if self.gts.is_Matrix:
            if self.lts.is_Matrix:
                diff = self.gts - self.lts
            else:
                if isinstance(self, (LMI_PD, LMI_PSD)):
                    return self  # self is already in canonical form
                else:
                    diff = self.gts
        else:
            diff = -self.lts

        diff = block_collapse(diff)

        if self.is_strict:
            return LMI_PD(diff, 0)
        else:
            return LMI_PSD(diff, 0)

    def expanded(self, variables):
        """Return the LMI as a sum of coefficent matrices times varibles form.
        """
        if self.lhs.is_Matrix:
            lhs = lm_sym_expanded(self.lhs, variables)
        else:
            lhs = self.lhs
        if self.rhs.is_Matrix:
            rhs = lm_sym_expanded(self.rhs, variables)
        else:
            rhs = self.rhs
        return self.func(lhs, rhs)

    def doit(self, **hints):
        if hints.get('deep', False):
            lhs = self.lhs.doit(**hints)
            rhs = self.lhs.doit(**hints)
            return self.func(lhs, rhs)
        else:
            return self


class LMI_PSD(BaseLMI, GreaterThan):
    """Class representation of Linear Matrix Inequality.

    Represents a non-stric LMI where left-hand side minus
    right-hand side (if any) is Positive Semi-Definite.

    Input matrices are checked for symmetry, pass `assert_symmetry=False`
    to force no symmetry assertion.

    Example:
    >>> from sympy import Matrix
    >>> from sympy.abc import x, y, z
    >>> from lmi_sdp import LMI_PSD
    >>> m = Matrix([[x, y], [y, z+1]])
    >>> LMI_PSD(m)
    Matrix([
    [x,     y],
    [y, z + 1]]) >= 0
    >>> m = Matrix([[x+y, y], [y, z]])
    >>> c = Matrix([[1, 2], [2, 3]])
    >>> LMI_PSD(m, c)
    Matrix([
    [x + y, y],
    [    y, z]]) >= Matrix([
    [1, 2],
    [2, 3]])
    """
    is_strict = False

    def __new__(cls, lhs, rhs=0, assert_symmetry=True):
        return BaseLMI.__new__(cls, lhs, rhs, GreaterThan, assert_symmetry)


class LMI_PD(BaseLMI, StrictGreaterThan):
    """Class representation of Linear Matrix Inequality.

    Represents a stric LMI where left-hand side minus
    right-hand side (if any) is Positive Definite.

    See LMI_PSD.__doc__ for common info
    """
    is_strict = True

    def __new__(cls, lhs, rhs=0, assert_symmetry=True):
        return BaseLMI.__new__(cls, lhs, rhs, StrictGreaterThan,
                               assert_symmetry)


class LMI_NSD(BaseLMI, LessThan):
    """Class representation of Linear Matrix Inequality.

    Represents a non-stric LMI where left-hand side minus
    right-hand side (if any) is Negative Semi-Definite.

    See LMI_PSD.__doc__ for common info
    """
    is_strict = False

    def __new__(cls, lhs, rhs=0, assert_symmetry=True):
        return BaseLMI.__new__(cls, lhs, rhs, LessThan, assert_symmetry)


class LMI_ND(BaseLMI, StrictLessThan):
    """Class representation of Linear Matrix Inequality.

    Represents a stric LMI where left-hand side minus
    right-hand side (if any) is Negative Definite.

    See LMI_PSD.__doc__ for common info
    """
    is_strict = True

    def __new__(cls, lhs, rhs=0, assert_symmetry=True):
        return BaseLMI.__new__(cls, lhs, rhs, StrictLessThan, assert_symmetry)

LMI = LMI_PSD  # default LMI type


def _print_BaseLMI(self, expr):
    charmap = {
        ">": r"\succ",
        "<": r"\prec",
        ">=": r"\succeq",
        "<=": r"\preceq",
    }
    return "%s %s %s" % (self._print(expr.lhs),
                         charmap[expr.rel_op], self._print(expr.rhs))


def init_lmi_latex_printing():
    """ Monkey patch SymPy LatexPrinter to include a BaseLMI printer"""
    from sympy.printing.latex import LatexPrinter
    LatexPrinter._print_BaseLMI = _print_BaseLMI
