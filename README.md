# DESincEig

The purpose of this `Julia` package is to provide an efficient fast general purpose differential eigenvalue solver package, supporting the canonical interval, and semi-infinite and infinite domains for Sturm-Liouville problems. The following algorithm utilizes the Double Exponential Sinc-Collocation method.

The primary function of this module computes the eigenvalues of a general Sturm-Liouville problem of the form:

```julia
(-D^2 + q(x) ) u(x) = \lambda \rho(x) u(x),  a < x < b,   with   u(a) = u(b) = 0,
where -\infty \leq a < b \leq \infty .

In the problem P1,
1. D is the differential operator, an
2. q(x) is a continuous bounded function defined on (a,b)
3. \rho(x) is a continuous positive function defined on (a,b)
4. u(x) are the eigenfunction corresponding to the eigenvalues \lambda. ```

```julia
using DESincEig
```
