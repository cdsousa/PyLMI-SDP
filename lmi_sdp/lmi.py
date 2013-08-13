"""LMI representation and tools"""

from sympy import sympify, GreaterThan, StrictGreaterThan, LessThan, \
    StrictLessThan

from sympy.matrices.matrices import MatrixError, NonSquareMatrixError, \
    ShapeError


class NonSymmetricMatrixError(ValueError, MatrixError):
    pass


class _LMI(object):
    """Not intended for general use

    _LMI is only used so that LMI_* classes can share functions.
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
            raise ValueError('LMI sides must two matrices or a matrix and a zero')

        return rel_cls.__new__(cls, lhs, rhs)

    def doit(self, **hints):
        if hints.get('deep', False):
            lhs = self.lhs.doit(**hints)
            rhs = self.lhs.doit(**hints)
            return self.func(lhs, rhs)
        else:
            return self


class LMI_PSD(_LMI, GreaterThan):
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
    def __new__(cls, lhs, rhs=0):
        return _LMI.__new__(cls, lhs, rhs, GreaterThan)


class LMI_PD(_LMI, StrictGreaterThan):
    """Class representation of Linear Matrix Inequality.

    Represents a stric LMI where left-hand side minus
    right-hand side (if any) is Positive Definite.
    """
    def __new__(cls, lhs, rhs=0):
        return _LMI.__new__(cls, lhs, rhs, StrictGreaterThan)


class LMI_NSD(_LMI, LessThan):
    """Class representation of Linear Matrix Inequality.

    Represents a non-stric LMI where left-hand side minus
    right-hand side (if any) is Negative Semi-Definite.
    """
    def __new__(cls, lhs, rhs=0):
        return _LMI.__new__(cls, lhs, rhs, LessThan)


class LMI_ND(_LMI, StrictLessThan):
    """Class representation of Linear Matrix Inequality.

    Represents a stric LMI where left-hand side minus
    right-hand side (if any) is Negative Definite.
    """
    def __new__(cls, lhs, rhs=0):
        return _LMI.__new__(cls, lhs, rhs, StrictLessThan)

LMI = LMI_PSD  # default LMI type
