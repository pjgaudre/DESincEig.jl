# DESincEig

The purpose of this `Julia` package is to provide an efficient fast general purpose differential eigenvalue solver package, supporting the canonical interval, and semi-infinite and infinite domains for Sturm-Liouville problems. The following algorithm utilizes the Double Exponential Sinc-Collocation method.

The primary function of this module computes the eigenvalues of a general Sturm-Liouville problem of the form:

```julia
P1 : (-D^2 + q(x) ) u(x) = ? ?(x) u(x),  a < x < b,   with   u(a) = u(b) = 0,     where -? ? a < b ? ? .
In the problem P1,
1. D is the differential operator, an
2. q(x) is a continuous bounded function defined on (a,b)
3. ?(x) is a continuous positive function defined on (a,b)
4. u(x) are the eigenfunction corresponding to the eigenvalues ?.
'''

In this algorithm, we use the transformation developed by Eggert et al. in reference [1]. This transformation results in a symmetric generalized eigenvalue problem defined as:

'''julia
P2 : (-D^2 + qtilde(x))v(x) = ? ?tilde(x) v(x),  -? < x < ?,  with   v(±?) = 0
In the problem P2,
1. D is the differential operator, an
2. qtilde(x) is the resulting transformed function defined on (-?,?)
3. ?tilde(x) is the resulting transformed function defined on (-?,?)
4. v(x) are the eigenfunction corresponding to the eigenvalues ?.
'''
 
Now, '''julia v(x)''' has double expoenential decay at both infinities. See reference [2] for more details of the form of '''julia qtilde(x)''' and '''julia ?tilde(x)'''.


To use this package, once simply writes:

```julia
using DESincEig
```
