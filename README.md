# DESincEig

The purpose of this `Julia` package is to provide an efficient fast general purpose differential eigenvalue solver package, supporting the canonical interval, and semi-infinite and infinite domains for Sturm-Liouville problems. The following algorithm utilizes the Double Exponential Sinc-Collocation method. This package allows the user to consider other domains by declaring a new instance of the type `Domain`.

The primary function of this module computes the eigenvalues of a general Sturm-Liouville problem of the form:
```julia
P1 : (-D^2 + q(x) ) u(x) = λ ρ(x) u(x),  a < x < b,   with   u(a) = u(b) = 0, 
      where -∞ ≦ a < b ≦ ∞  .
In the problem P1,
1. D is the differential operator, an
2. q(x) is a continuous bounded function defined on (a,b)
3. ρ(x) is a continuous positive function defined on (a,b)
4. u(x) are the eigenfunction corresponding to the eigenvalues λ.
```
In this algorithm, we use the transformation developed by Eggert et al. in reference <a href="http://dx.doi.org/10.1016/0021-9991(87)90163-X">[1]</a>. This transformation results in a symmetric generalized eigenvalue problem defined as:
```julia
P2 : (-D^2 + qtilde(x))v(x) = λ ρtilde(x) v(x),  -∞ < x < ∞,  with   v(±∞) = 0
In the problem P2,
1. D is the differential operator, an
2. qtilde(x) is the resulting transformed function defined on (-∞,∞)
3. ρtilde(x) is the resulting transformed function defined on (-∞,∞)
4. v(x) are the eigenfunction corresponding to the eigenvalues λ.
```
Now, ```v(x)``` has double expoenential decay at both infinities. See reference <a href="http://arxiv.org/abs/1409.7471v3">[2]</a> for more details of the form of ```qtilde(x)``` and ```ρtilde(x)```.

To use this package, once simply writes:
```julia
using DESincEig
```
### Example 1 from <a href="http://dx.doi.org/10.1016/j.aop.2015.05.026">[5]</a>

Suppose we are interested in computing the energy eigenvalues ```E``` of Schrödinger equation:

```julia
(-D^2 + V(x) ) ψ(x) = E ψ(x),  with  ψ(±∞) = 0
```
for the potential:
```julia
V(x) = x^2 + x^4,
```
We use the package function `SincEigen` to calculate the Eigenvalues:

```julia
gamma = 2.0
Beta = (0.5)^gamma / gamma
dopt = pi/2gamma
SincEigen(x->x.^2+x.^4,ones,Infinite2SL,[Beta,Beta],[gamma,gamma],dopt)
```

# References:

1.  N. Eggert, M. Jarratt, and J. Lund. <a href="http://dx.doi.org/10.1016/0021-9991(87)90163-X"> Sinc function computation of the eigenvalues of Sturm-Liouville problems</a>. SIAM Journal of Computational Physics, 69:209-229, 1987
2.  P. Gaudreau, R.M. Slevinsky, and H. Safouhi. <a href="http://arxiv.org/abs/1409.7471v3"> The Double Exponential Sinc Collocation Method for Singular Sturm-Liouville Problems</a>. arXiv:1409.7471v2, 2014
3.  R. M. Slevinsky and S. Olver. <a href="http://dx.doi.org/10.1137/140978363">On the use of conformal maps for the acceleration of convergence of the trapezoidal rule and Sinc numerical methods</a>, SIAM J. Sci. Comput., 37:A676--A700, 2015. An earlier version appears here: <a href="http://arxiv.org/abs/1406.3320"> arXiv:1406.3320</a>.
4.  R.M. Corless, G.H. Gonnet, D.E.G. Hare, D.J. Jeffrey, D.E. Knuth, <a href="http://link.springer.com/article/10.1007%2FBF02124750"> On the Lambert W function</a>. Advances in Computational Mathematics, 5(1):329-359, 1996.
5.  P. Gaudreau, R.M. Slevinsky, and H. Safouhi, <a href="http://dx.doi.org/10.1016/j.aop.2015.05.026"> Computing energy eigenvalues of anharmonic oscillators using the double exponential Sinc collocation method</a>, Annals of Physics, 360:520-538, 2015.



