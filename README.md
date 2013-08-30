PyLMI-SDP
=========

*Symbolic linear matrix inequalities (LMI) and semi-definite programming (SDP) tools for Python*

This package includes a set of classes to represent and manipulate LMIs symbolically using [SymPy](http://sympy.org).
It also includes tools to export LMIs to [CVXOPT](http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#semidefinite-programming) SDP input and to the [SDPA](http://sdpa.sourceforge.net/) format.

Depends on [SymPy](http://sympy.org) and [NumPy](http://www.numpy.org/), and optionally on [CVXOPT](http://cvxopt.org/).

Examples
--------

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
\left[\begin{smallmatrix}{}x + 1 & y + 2\\y + 2 & x + z\end{smallmatrix}\right] \succ 0

```
![equation](http://latex.codecogs.com/gif.latex?%5Cleft%5B%5Cbegin%7Bsmallmatrix%7D%7B%7Dx%2B1%26y%2B2%5C%5Cy%2B2%26x%2Bz%5Cend%7Bsmallmatrix%7D%5Cright%5D%5Csucc0)

```Python
>>> print(latex(lmi.expanded(variables)))
\left[\begin{smallmatrix}{}1.0 & 0.0\\0.0 & 1.0\end{smallmatrix}\right] x + \left[\begin{smallmatrix}{}0.0 & 1.0\\1.0 & 0.0\end{smallmatrix}\right] y + \left[\begin{smallmatrix}{}0.0 & 0.0\\0.0 & 1.0\end{smallmatrix}\right] z + \left[\begin{smallmatrix}{}1.0 & 2.0\\2.0 & 0.0\end{smallmatrix}\right] \succ 0

```
![equation](http://latex.codecogs.com/gif.latex?%5Cleft%5B%5Cbegin%7Bsmallmatrix%7D%7B%7D1.0%260.0%5C%5C0.0%261.0%5Cend%7Bsmallmatrix%7D%5Cright%5Dx%2B%5Cleft%5B%5Cbegin%7Bsmallmatrix%7D%7B%7D0.0%261.0%5C%5C1.0%260.0%5Cend%7Bsmallmatrix%7D%5Cright%5Dy%2B%5Cleft%5B%5Cbegin%7Bsmallmatrix%7D%7B%7D0.0%260.0%5C%5C0.0%261.0%5Cend%7Bsmallmatrix%7D%5Cright%5Dz%2B%5Cleft%5B%5Cbegin%7Bsmallmatrix%7D%7B%7D1.0%262.0%5C%5C2.0%260.0%5Cend%7Bsmallmatrix%7D%5Cright%5D%5Csucc0)

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
\left[\begin{smallmatrix}{}- x & - y\\- y & - x - z\end{smallmatrix}\right] \preceq \left[\begin{smallmatrix}{}1 & 2\\2 & 0\end{smallmatrix}\right]

```
![equation](http://latex.codecogs.com/gif.latex?%5Cleft%5B%5Cbegin%7Bsmallmatrix%7D%7B%7D-x%26-y%5C%5C-y%26-x-z%5Cend%7Bsmallmatrix%7D%5Cright%5D%5Cpreceq%5Cleft%5B%5Cbegin%7Bsmallmatrix%7D%7B%7D1%262%5C%5C2%260%5Cend%7Bsmallmatrix%7D%5Cright%5D)


Author
------

[Cristóvão Duarte Sousa](https://github.com/cdsousa)

Install
-------

From git source:

    git clone git@github.com:cdsousa/PyLMI-SDP.git
    cd PyLMI-SDP
    python setup.py install

License
-------

Simplified BSD License. See [License File](LICENSE.txt)
