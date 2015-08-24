# DESincEig

The purpose of this `Julia` package is to provide an efficient fast general purpose differential eigenvalue solver package, supporting the canonical interval, and semi-infinite and infinite domains for Sturm-Liouville problems. The following algorithm utilizes the Double Exponential Sinc-Collocation method. This package allows the user to consider other domains by declaring a new instance of the type `DomainSL`.

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

The type `DomainSL` is used to select from the conformal maps depending on the domain of the problem and the decay rate of the solution of P1.
1. For S-L problems on a finite domain `I=(a,b)` with algebraic decay at the endpoints:
`DomainSL = FiniteSL`.
2. For S-L problems on a infinite domain `I=(-∞,∞)` with algebraic decay at the endpoints:
`DomainSL = Infinite1SL`.
3. For S-L problems on a infinite domain `I=(-∞,∞)` with single-exponential decay at the endpoints:
`DomainSL = Infinite2SL`.
4. For S-L problems on a semi-infinite domain `I=(0,∞)` with single-exponential decay at infinity and algebraic decay at 0:
`DomainSL = SemiInfiniteSL`.

To use this package, once simply writes:
```julia
using DESincEig
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
`S = { z ∈ C : qtilde(z) or ρtilde(z) does not exist a.}`
let s denote the positive imaginary part of nearest singularity to the real axis:
`s = min | Im{S} |`,
then `d = min{ π/2max{γL,γR} , s }`.
```Julia
Input:
Necessary parameters
1. q(x):: Function,      The function in P1
2. ρ(x):: Function,      The function in P1
3. domain:: DomainSL,    FiniteSL, Infinite1SL, Infinite2SL or SemiInfiniteSL
4. βopt:: Vector{T},     [βL,βR]
5. γopt:: Vector{T},     [γL,γR]
6. d:: Number,           min{ π/2max{γL,γR} , s }
____________________________________________________________________________________________________________________
Extra parameters ( not necessary but can offer more options )
7. enum::Vector{T},         [n,tv]: Default is [NaN,NaN]
                            n ∈ {0,1,2,3,...} :  Eigenvalue number
                            tv = λ_{n} : True Value for eigenvalue λ_{n}
                            If specified, the absolute error is computed instead of the absolute error approximation
                            of the DESCM w.r.t. to λ_{n} will be returned.

8. tol::T,                  tolerance level: Default is 5e-12
                            tol is used to find the optimal Matrix Size in order for the approximate eigenvalue
                            μ to have an an aprroximation to the absolute error less than tol.

9. Range::Vector{Integer},  [start:skip:end]: Default = [1:1:100]
                            Will run algorithm with N or M = "start" with index jump "skip" until "end".

10. u0::T                   Parameter used in the inner map construction H(t) presented above.
                            Default = one(T)

11. u::Vector{T}            Parameters used in the inner map construction H(t) presented above.
                            Default = [zero(T)]

12. Trace_Mesh::Bool         Default = false
If true, the mesh size will be computed by minimizing the trace of the matrix D^(-1)HD^(-1).
resulting from the DESCM. For even functions q(x) and ρ(x) and an infinite domain: DomainSL = Infinite1SL or Infinite2SL,
once can minimize this functional to obtain an alternate mesh size: htilde. This alternate mesh-size has proven
to be better suited for highly-oscillatory functions q(x). This functional is minimized using the Julia package: Optim.
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

1.  N. Eggert, M. Jarratt, and J. Lund. <a href="http://dx.doi.org/10.1016/0021-9991(87)90163-X"> Sinc function computation       of the eigenvalues of Sturm-Liouville problems</a>. SIAM Journal of Computational Physics, 69:209-229, 1987
2.  P. Gaudreau, R.M. Slevinsky, and H. Safouhi. <a href="http://arxiv.org/abs/1409.7471v3"> The Double Exponential Sinc       Collocation Method for Singular Sturm-Liouville Problems</a>. arXiv:1409.7471v2, 2014
3.  R. M. Slevinsky and S. Olver. <a href="http://dx.doi.org/10.1137/140978363">On the use of conformal maps for the               acceleration of convergence of the trapezoidal rule and Sinc numerical methods</a>, SIAM J. Sci. Comput.,                    37:A676--A700, 2015. An earlier version appears here: <a href="http://arxiv.org/abs/1406.3320"> arXiv:1406.3320</a>.
4.  R.M. Corless, G.H. Gonnet, D.E.G. Hare, D.J. Jeffrey, D.E. Knuth, <a                                                           href="http://link.springer.com/article/10.1007%2FBF02124750"> On the Lambert W function</a>. Advances in Computational       Mathematics, 5(1):329-359, 1996.
5.  P. Gaudreau, R.M. Slevinsky, and H. Safouhi, <a href="http://dx.doi.org/10.1016/j.aop.2015.05.026"> Computing energy           eigenvalues of anharmonic oscillators using the double exponential Sinc collocation method</a>, Annals of Physics,           360:520-538, 2015.



