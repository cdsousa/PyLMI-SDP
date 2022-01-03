PyLMI-SDP
=========

*Symbolic linear matrix inequalities (LMI) and semi-definite programming (SDP) tools for Python*

This package includes a set of classes to represent and manipulate LMIs symbolically using [SymPy](http://sympy.org).
It also includes tools to export LMIs to [CVXOPT](http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#semidefinite-programming) SDP input and to the [SDPA](http://sdpa.sourceforge.net/) format.

Depends on [SymPy](http://sympy.org) 0.7.3 and on [NumPy](http://www.numpy.org/) 1.7.1; and optionally on [CVXOPT](http://cvxopt.org/) and on [SciPy](http://www.scipy.org/) (for sparse matrices).
Single codebase supporting both Python 2.7 and Python 3.3.
PyLMI-SDP is tested in these versions but it may work in others.

PyLMI-SDP is at [GitHub](https://github.com/cdsousa/PyLMI-SDP)

[![Build Status](https://travis-ci.org/cdsousa/PyLMI-SDP.png?branch=master)](https://travis-ci.org/cdsousa/PyLMI-SDP)
[![Coverage Status](https://coveralls.io/repos/cdsousa/PyLMI-SDP/badge.png?branch=master)](https://coveralls.io/r/cdsousa/PyLMI-SDP?branch=master)

LMI Definition
--------------

### Examples

```Python
>>> from sympy import symbols, Matrix
>>> from lmi_sdp import LMI_PD, LMI_NSD
>>> variables = symbols('x y z')
>>> x, y, z = variables
>>> lmi = LMI_PD(Matrix([[x+1, y+2], [y+2, z+x]]))
>>> lmi
Matrix([
[x + 1, y + 2],
[y + 2, x + z]]) > 0

```

```Python
>>> from lmi_sdp import init_lmi_latex_printing
>>> from sympy import latex
>>> init_lmi_latex_printing()
>>> print(latex(lmi))
\left[\begin{matrix}x + 1 & y + 2\\y + 2 & x + z\end{matrix}\right] \succ 0

```
![lmi](http://latex.codecogs.com/gif.latex?%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7Dx%2B1%26y%2B2%5C%5Cy%2B2%26x%2Bz%5Cend%7Bmatrix%7D%5Cright%5D%5Csucc0)

```Python
>>> print(latex(lmi.expanded(variables)))
\left[\begin{matrix}1.0 & 0.0\\0.0 & 1.0\end{matrix}\right] x + \left[\begin{matrix}0.0 & 1.0\\1.0 & 0.0\end{matrix}\right] y + \left[\begin{matrix}0.0 & 0.0\\0.0 & 1.0\end{matrix}\right] z + \left[\begin{matrix}1.0 & 2.0\\2.0 & 0.0\end{matrix}\right] \succ 0

```
![lmi.expanded(variables)](http://latex.codecogs.com/gif.latex?%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D1.0%260.0%5C%5C0.0%261.0%5Cend%7Bmatrix%7D%5Cright%5Dx%2B%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D0.0%261.0%5C%5C1.0%260.0%5Cend%7Bmatrix%7D%5Cright%5Dy%2B%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D0.0%260.0%5C%5C0.0%261.0%5Cend%7Bmatrix%7D%5Cright%5Dz%2B%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D1.0%262.0%5C%5C2.0%260.0%5Cend%7Bmatrix%7D%5Cright%5D%5Csucc0)

```Python
>>> lmi_2 = LMI_NSD( Matrix([[-x, -y], [-y, -z-x]]), Matrix([[1, 2], [2, 0]]))
>>> lmi_2
Matrix([
[-x,     -y],
[-y, -x - z]]) <= Matrix([
[1, 2],
[2, 0]])
>>> lmi_2.canonical()
Matrix([
[x + 1, y + 2],
[y + 2, x + z]]) >= 0

```

```Python
>>> print(latex(lmi_2))
\left[\begin{matrix}- x & - y\\- y & - x - z\end{matrix}\right] \preceq \left[\begin{matrix}1 & 2\\2 & 0\end{matrix}\right]

```
![lmi_2](http://latex.codecogs.com/gif.latex?%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D-x%26-y%5C%5C-y%26-x-z%5Cend%7Bmatrix%7D%5Cright%5D%5Cpreceq%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D1%262%5C%5C2%260%5Cend%7Bmatrix%7D%5Cright%5D)

Convertion to CVXOPT SDP
------------------------

### Example

(from CVXOPT [SDP example](http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#semidefinite-programming))

```Python
>>> from sympy import symbols, Matrix
>>> from lmi_sdp import LMI_NSD, init_lmi_latex_printing
>>>
>>> init_lmi_latex_printing()
>>>
>>> variables = symbols('x1 x2 x3')
>>> x1, x2, x3 = variables
>>>
>>> min_obj = x1 - x2 + x3
>>>
>>> LMI_1 = LMI_NSD(
...     x1*Matrix([[-7, -11], [-11, 3]]) +
...     x2*Matrix([[7, -18], [-18, 8]]) +
...     x3*Matrix([[-2, -8], [-8, 1]]),
...     Matrix([[33, -9], [-9, 26]]))
>>>
>>> LMI_2 = LMI_NSD(
...     x1*Matrix([[-21, -11, 0], [-11, 10, 8], [0, 8, 5]]) +
...     x2*Matrix([[0, 10, 16], [10, -10, -10], [16, -10, 3]]) +
...     x3*Matrix([[-5, 2, -17], [2, -6, 8], [-17, 8, 6]]),
...     Matrix([[14, 9, 40], [9, 91, 10], [40, 10, 15]]))
>>>
>>> min_obj
x1 - x2 + x3

```
![min_obj](http://latex.codecogs.com/gif.latex?x_%7B1%7D-x_%7B2%7D%2Bx_%7B3%7D)

```Python
>>> LMI_1.expanded(variables)
Matrix([
[ -7.0, -11.0],
[-11.0,   3.0]])*x1 + Matrix([
[  7.0, -18.0],
[-18.0,   8.0]])*x2 + Matrix([
[-2.0, -8.0],
[-8.0,  1.0]])*x3 <= Matrix([
[33, -9],
[-9, 26]])

```
![LMI_1.expanded(variables)](http://latex.codecogs.com/gif.latex?%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D-7.0%26-11.0%5C%5C-11.0%263.0%5Cend%7Bmatrix%7D%5Cright%5Dx_%7B1%7D%2B%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D7.0%26-18.0%5C%5C-18.0%268.0%5Cend%7Bmatrix%7D%5Cright%5Dx_%7B2%7D%2B%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D-2.0%26-8.0%5C%5C-8.0%261.0%5Cend%7Bmatrix%7D%5Cright%5Dx_%7B3%7D%5Cpreceq%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D33%26-9%5C%5C-9%2626%5Cend%7Bmatrix%7D%5Cright%5D)

```Python
>>> LMI_2.expanded(variables)
Matrix([
[-21.0, -11.0, 0.0],
[-11.0,  10.0, 8.0],
[  0.0,   8.0, 5.0]])*x1 + Matrix([
[ 0.0,  10.0,  16.0],
[10.0, -10.0, -10.0],
[16.0, -10.0,   3.0]])*x2 + Matrix([
[ -5.0,  2.0, -17.0],
[  2.0, -6.0,   8.0],
[-17.0,  8.0,   6.0]])*x3 <= Matrix([
[14,  9, 40],
[ 9, 91, 10],
[40, 10, 15]])

```
![LMI_2.expanded(variables)](http://latex.codecogs.com/gif.latex?%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D-21.0%26-11.0%260.0%5C%5C-11.0%2610.0%268.0%5C%5C0.0%268.0%265.0%5Cend%7Bmatrix%7D%5Cright%5Dx_%7B1%7D%2B%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D0.0%2610.0%2616.0%5C%5C10.0%26-10.0%26-10.0%5C%5C16.0%26-10.0%263.0%5Cend%7Bmatrix%7D%5Cright%5Dx_%7B2%7D%2B%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D-5.0%262.0%26-17.0%5C%5C2.0%26-6.0%268.0%5C%5C-17.0%268.0%266.0%5Cend%7Bmatrix%7D%5Cright%5Dx_%7B3%7D%5Cpreceq%5Cleft%5B%5Cbegin%7Bmatrix%7D%7B%7D14%269%2640%5C%5C9%2691%2610%5C%5C40%2610%2615%5Cend%7Bmatrix%7D%5Cright%5D)

```Python
>>> from cvxopt import solvers
>>> from lmi_sdp import to_cvxopt
>>>
>>> solvers.options['show_progress'] = False
>>>
>>> c, Gs, hs = to_cvxopt(min_obj, [LMI_1, LMI_2], variables)
>>>
>>> sol = solvers.sdp(c, Gs=Gs, hs=hs)
>>> print(sol['x'])
[-3.68e-01]
[ 1.90e+00]
[-8.88e-01]
<BLANKLINE>

```

Export to SDPA Format
---------------------

### Example

```Python
>>> from sympy import symbols, Matrix
>>> from lmi_sdp import LMI_PSD, to_sdpa_sparse
>>>
>>> variables = x1, x2 = symbols('x1 x2')
>>>
>>> min_obj = 10*x1 + 20*x2
>>> lmi_1 = LMI_PSD(
...     -Matrix([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]) +
...     Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])*x1 +
...     Matrix([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 2], [0, 0, 2, 6]])*x2)
>>> lmi_1
Matrix([
[x1 - 1,           0,        0,        0],
[     0, x1 + x2 - 2,        0,        0],
[     0,           0, 5*x2 - 3,     2*x2],
[     0,           0,     2*x2, 6*x2 - 4]]) >= 0
>>>
>>> dat = to_sdpa_sparse(min_obj, lmi_1, variables, comment='test sparse')
>>> print(dat)
"test sparse"
2 = ndim
3 = nblocks
1 1 2 = blockstruct
10.0, 20.0 = objcoeffs
0 1 1 1 1.0
0 2 1 1 2.0
0 3 1 1 3.0
0 3 2 2 4.0
1 1 1 1 1.0
1 2 1 1 1.0
2 2 1 1 1.0
2 3 1 1 5.0
2 3 1 2 2.0
2 3 2 2 6.0
<BLANKLINE>

```


Author
------

[Cristóvão Duarte Sousa](https://github.com/cdsousa)

Install
-------

From git source:

    git clone https://github.com/cdsousa/PyLMI-SDP.git
    cd PyLMI-SDP
    python setup.py install

License
-------

Simplified BSD License. See [License File](LICENSE.txt)
