# DESincEig.jl

The purpose of this `Julia` package is to provide an efficient fast general purpose differential eigenvalue solver package, supporting the canonical interval, and semi-infinite and infinite domains for Sturm-Liouville problems. The following algorithm utilizes the Double Exponential Sinc-Collocation method. This package allows the user to consider other domains by declaring a new instance of the `SincFun` type `Domain`.

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
In this algorithm, we use the transformation developed by Eggert et al. in reference <a href="http://dx.doi.org/10.1016/0021-9991(87)90163-X">[1]</a>.
This transformation results in a symmetric generalized eigenvalue problem defined as:
```julia
P2 : (-D^2 + qtilde(x))v(x) = λ ρtilde(x) v(x),  -∞ < x < ∞,  with   v(±∞) = 0

In the problem P2,
1. D is the differential operator, an
2. qtilde(x) is the resulting transformed function defined on (-∞,∞)
3. ρtilde(x) is the resulting transformed function defined on (-∞,∞)
4. v(x) are the eigenfunction corresponding to the eigenvalues λ.
   v(x) now have double expoenential decay at both infinities.
 ```
See reference <a href="http://arxiv.org/abs/1409.7471v3">[2]</a> for more details of the form of `qtilde(x)` and `ρtilde(x)`.

The type `Domain` is used to select from the conformal maps depending on the domain of the problem and the decay rate of the solution of P1.
1. For S-L problems on a finite domain `I=(a,b)` with algebraic decay at the endpoints:
`Domain = Finite(a,b)`.

2. For S-L problems on a infinite domain `I=(-∞,∞)` with algebraic decay at the endpoints:
`Domain = Infinite1{Float64}()`.

3. For S-L problems on a infinite domain `I=(-∞,∞)` with single-exponential decay at the endpoints:
`Domain = Infinite2{Float64}()`.

4. For S-L problems on a semi-infinite domain `I=(0,∞)` with single-exponential decay at infinity and algebraic decay at 0:
`Domain = SemiInfinite1{Float64}()`.

To use this package, once simply writes:
```julia
using SincFun, DESincEig
```

The main function of this package is `SincEigen`. This function computes the eigenvalues of the the Sturm-Liouville (P1) using the double exponential Sinc-Collocation method.
For optimal results, it is worth performing an asymptotic analysis of the solution of P2.
The resulting analysis will lead to the following bounds for constants `βL,βR,γL,γR>0`:
```Julia
 |v(x)| < AL exp(-βL exp ( γL |x| ) ) , on (-∞,0)
 |v(x)| < AR exp(-βR exp ( γR |x| ) ) , on (0, ∞)
 ```
The parameters `βL, βR, γL, γR` are used in the calculation of the mesh size, `h`, for the DESCM.

Another important parameter involved in the calculations of the mesh size for the DESCM is the width, `d`, of the strip `D_{d}`.
Let `S` denote the set of complex singularities of the functions `qtilde(x)` and `ρtilde(x)`:
`S = { z ∈ C : qtilde(z) or ρtilde(z) does not exist.}`
let s denote the positive imaginary part of nearest singularity to the real axis:
`s = min | Im{S} |`,
then `d = min{ π/2max{γL,γR} , s }`.
```Julia
Input:
Necessary parameters
1. q(x):: Function,      The function in P1
2. ρ(x):: Function,      The function in P1
3. domain:: Domain,    Finite(a,b), Infinite1{Float64}(), Infinite2{Float64}() or SemiInfinite1{Float64}()
4. βopt:: Vector{T},     [βL,βR]
5. γopt:: Vector{T},     [γL,γR]
6. d:: Number,           min{ π/2max{γL,γR} , s }
```


### Example 1 from <a href="http://arxiv.org/abs/1409.7471v3">[2]</a>
Suppose we are interested in computing the eigenvalues of the Laguerre equation:
```julia
(-D^2 + (35/4)/x^2  - 2 + x^2 /16  ) u(x) = λ u(x),  with u(0) = u(∞) = 0
```
we use the package function `SincEigen` to calculate the eigenvalues:
```julia
SincEigen( x -> (35/4)./x.^2  .- 2 .+ x.^2 /16 , ones , SemiInfinite1{Float64}() , [1.5,0.03125] , [1.0,2.0] , pi/4 )
```

### Example 2 from <a href="http://dx.doi.org/10.1016/j.aop.2015.05.026">[5]</a>
Suppose we are interested in computing the energy eigenvalues ```E``` of Schrödinger equation:
```julia
(-D^2 + V(x) ) ψ(x) = E ψ(x),  with  ψ(±∞) = 0
```
for the quartic anharmonic oscillator potential:
```julia
V(x) = x.^2 + x.^4
```
we use the package function `SincEigen` to calculate the eigenvalues:
```julia
SincEigen(V, ones, Infinite2{Float64}(), [0.125,0.125] , [2.0,2.0] , pi/4 )
```

# References:

1.  N. Eggert, M. Jarratt, and J. Lund. <a href="http://dx.doi.org/10.1016/0021-9991(87)90163-X"> Sinc function computation       of the eigenvalues of Sturm-Liouville problems</a>. SIAM Journal of Computational Physics, 69:209-229, 1987
2.  P. Gaudreau, R.M. Slevinsky, and H. Safouhi. <a href="http://arxiv.org/abs/1409.7471v3"> The Double Exponential Sinc       Collocation Method for Singular Sturm-Liouville Problems</a>. arXiv:1409.7471v2, 2014
3.  R. M. Slevinsky and S. Olver. <a href="http://dx.doi.org/10.1137/140978363">On the use of conformal maps for the               acceleration of convergence of the trapezoidal rule and Sinc numerical methods</a>, SIAM J. Sci. Comput.,                    37:A676--A700, 2015. An earlier version appears here: <a href="http://arxiv.org/abs/1406.3320"> arXiv:1406.3320</a>.
4.  R.M. Corless, G.H. Gonnet, D.E.G. Hare, D.J. Jeffrey, D.E. Knuth, <a                                                           href="http://link.springer.com/article/10.1007%2FBF02124750"> On the Lambert W function</a>. Advances in Computational       Mathematics, 5(1):329-359, 1996.
5.  P. Gaudreau, R.M. Slevinsky, and H. Safouhi, <a href="http://dx.doi.org/10.1016/j.aop.2015.05.026"> Computing energy           eigenvalues of anharmonic oscillators using the double exponential Sinc collocation method</a>, Annals of Physics,           360:520-538, 2015.
6.   P. Gaudreau, and H. Safouhi.  <a href="http://arxiv.org/abs/1507.06709">Centrosymmetric Matrices in the Sinc Collocation Method for Sturm-Liouville Problems</a>. arXiv:1507.06709v1, 2015


