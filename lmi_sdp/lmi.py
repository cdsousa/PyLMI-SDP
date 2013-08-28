"""LMI representation and tools"""

from sympy import sympify, GreaterThan, StrictGreaterThan, LessThan, \
    StrictLessThan

from sympy.matrices.matrices import MatrixError, NonSquareMatrixError, \
    ShapeError


class NonSymmetricMatrixError(ValueError, MatrixError):
    pass


class BaseLMI(object):
    """Not intended for general use

    BaseLMI is used so that LMI_* classes can share functions.
    """
    def __new__(cls, lhs, rhs, rel_cls):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        if lhs.is_Matrix and not lhs.is_symmetric():
                raise NonSymmetricMatrixError('lhs matrix is not symmetric')
        if rhs.is_Matrix and not rhs.is_symmetric():
                raise NonSymmetricMatrixError('rsh matrix is not symmetric')
        if lhs.is_Matrix and rhs.is_Matrix:
                if lhs.shape != rhs.shape:
                    raise ShapeError('LMI matrices have different shapes')
        elif not ((lhs.is_Matrix and rhs.is_zero)
                  or (lhs.is_zero and rhs.is_Matrix)):
            raise ValueError('LMI sides must be two matrices '
                             'or a matrix and a zero')

        return rel_cls.__new__(cls, lhs, rhs)

    def canonical(self):
        """Returns the LMI positive (semi-)definite form with the matrix at
        the rhs and zero at the lhs.
        """
        if self.gts.is_Matrix:
            if self.lts.is_Matrix:
                diff = self.gts - self.lts
            else:
                diff = self.gts
        else:
            diff = -self.lts

        if self.is_strict:
            return LMI_PD(diff, 0)
        else:
            return LMI_PSD(diff, 0)

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

    def __new__(cls, lhs, rhs=0):
        return BaseLMI.__new__(cls, lhs, rhs, GreaterThan)


class LMI_PD(BaseLMI, StrictGreaterThan):
    """Class representation of Linear Matrix Inequality.

    Represents a stric LMI where left-hand side minus
    right-hand side (if any) is Positive Definite.
    """
    is_strict = True

    def __new__(cls, lhs, rhs=0):
        return BaseLMI.__new__(cls, lhs, rhs, StrictGreaterThan)


class LMI_NSD(BaseLMI, LessThan):
    """Class representation of Linear Matrix Inequality.

    Represents a non-stric LMI where left-hand side minus
    right-hand side (if any) is Negative Semi-Definite.
    """
    is_strict = False

    def __new__(cls, lhs, rhs=0):
        return BaseLMI.__new__(cls, lhs, rhs, LessThan)


class LMI_ND(BaseLMI, StrictLessThan):
    """Class representation of Linear Matrix Inequality.

    Represents a stric LMI where left-hand side minus
    right-hand side (if any) is Negative Definite.
    """
    is_strict = True

    def __new__(cls, lhs, rhs=0):
        return BaseLMI.__new__(cls, lhs, rhs, StrictLessThan)

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
    from sympy.printing.latex import LatexPrinter
    LatexPrinter._print_BaseLMI = _print_BaseLMI
