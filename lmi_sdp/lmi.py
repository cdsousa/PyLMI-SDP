"""LMI representation and tools"""

from sympy import sympify, GreaterThan, StrictGreaterThan, LessThan, \
    StrictLessThan


class LMI_PSD(GreaterThan):
    """Class representation of Linear Matrix Inequality.

    Represents a non-stric LMI using sympy GreaterThan subclass, where
    left-hand side minus right-hand side (if any) is Positive Semi-Definite.

    Example:
    >>> from sympy import Matrix
    >>> from sympy.abc import x, y, z
    >>> from lmi_sdp import LMI_PSD
    >>> m = Matrix([[x, y], [y, z+1]])
    >>> print(LMI_PSD(m))
    [x,     y]
    [y, z + 1] >= 0
    >>> m = Matrix([[x+y, y], [y, z]])
    >>> c = Matrix([[1, 2], [2, 3]])
    >>> print(LMI_PSD(m, c))
    [x + y, y]
    [    y, z] >= [1, 2]
    [2, 3]
    """
    def __new__(cls, lhs, rhs=0):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        return super(LMI_PSD, cls).__new__(cls, lhs, rhs)

    def doit(self):
        return self


class LMI_PD(StrictGreaterThan):
    """Class representation of Linear Matrix Inequality.

    Represents a stric LMI using sympy StrictGreaterThan subclass, where
    left-hand side minus right-hand side (if any) is Positive Definite.
    """
    def __new__(cls, lhs, rhs=0):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        return super(LMI_PD, cls).__new__(cls, lhs, rhs)

    def doit(self):
        return self


class LMI_NSD(LessThan):
    """Class representation of Linear Matrix Inequality.

    Represents a non-stric LMI using sympy LessThan subclass, where
    left-hand side minus right-hand side (if any) is Negative Semi-Definite.
    """
    def __new__(cls, lhs, rhs=0):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        return super(LMI_NSD, cls).__new__(cls, lhs, rhs)

    def doit(self):
        return self


class LMI_ND(StrictLessThan):
    """Class representation of Linear Matrix Inequality.

    Represents a stric LMI using sympy StrictLessThan subclass, where
    left-hand side minus right-hand side (if any) is Negative Definite.
    """
    def __new__(cls, lhs, rhs=0):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        return super(LMI_ND, cls).__new__(cls, lhs, rhs)

    def doit(self):
        return self

LMI = LMI_PSD
