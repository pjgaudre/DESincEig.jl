module DESincEig
#=
A module by Philippe Gaudreau,
Department of Mathematical & Statistical Sciences,
University of Alberta, 2014.

The primary function of this module (function SincEigen) computes the eigenvalues of
Singular Sturm-Liouviles problems using the DESINC Method.
_________________________________________________________________________________
A general S-L problem has the following form:

P1 : (-D^2 + q(x) ) u(x) = λ ρ(x) u(x),  a < x < b,   with   u(a) = u(b) = 0,
     where -∞ ≦ a < b ≦ ∞ .

In the problem P1,
1. D is the differential operator, an
2. q(x) is a continuous bounded function defined on (a,b)
3. ρ(x) is a continuous positive function defined on (a,b)
4. u(x) are the eigenfunction corresponding to the eigenvalues λ.
_________________________________________________________________________________

In this algorithm, we use the transformation developed by Eggert et al. in reference [1].
This transformation results in a symmetric generalized eigenvalue problem defined as:

P2 : (-D^2 + qtilde(x))v(x) = λ ρtilde(x) v(x),  -∞ < x < ∞,  with   v(±∞) = 0

In the problem P2,
1. D is the differential operator, an
2. qtilde(x) is the resulting transformed function defined on (-∞,∞)
3. ρtilde(x) is the resulting transformed function defined on (-∞,∞)
4. v(x) are the eigenfunction corresponding to the eigenvalues λ.
   v(x) now have double expoenential decay at both infinities.
See reference [2] for more details of the form of qtilde(x) and ρtilde(x)
_________________________________________________________________________________


References:
  1.  N. Eggert, M. Jarratt, and J. Lund. Sinc function computation of the eigenvalues of
              Sturm-Liouville problems. SIAM Journal of Computational Physics,
              69:209-229, 1987
  2.  P. Gaudreau, R.M. Slevinsky, and H. Safouhi. The Double Exponential Sinc Collocation
              Method for Singular Sturm-Liouville Problems. arXiv:1409.7471v2, 2014
  3.  R.M. Slevinsky and S. Olver. On the use of conformal maps for the acceleration of
              convergence of the trapezoidal rule and Sinc numerical methods.
              arXiv:1406:3320, 2014
  4.  R.M. Corless, G.H. Gonnet, D.E.G. Hare, D.J. Jeffrey, D.E. Knuth, On the Lambert W function.
              Advances in Computational Mathematics, 5(1):329-359, 1996
=#

include("Sinc.jl")
using Optim
export SincEigen, DomainSL , SincEigenStop , lambert_W
export FiniteSL,Infinite1SL,Infinite2SL,SemiInfiniteSL

####################### Conformal mappings based on different domains #####################################
#=
In this first section of the DESincEig module, we will construct various conformal mappings to induce
a double expoential decay at both infinities for the solution of the transformed S-L problem.

These conformal mappings are based on the domain of the orginal problem ( see problem P1 above )
as well as the type of decay of the solution u(x) at the end-points of its domain.

The conformal mappings, ϕ(t), will be created as a composition of an outer map, ψ(t), with an inner map, H(t).
In other words, the conformal mappings have the form:

ϕ(t) = (ψ∘H)(t) = ψ(H(t)),
_____________________________________ DEFINITION OF OUTER MAPS________________________________________
The type DomainSL is used to select from the outer maps depending on the domain of the problem.
Each element of type DomainSL is comprised of three functions:
 1. ψ  : The outer map
 2. ψp : The derivative of ψ, (ψ prime)
 3. ψtilde : The first half of the resulting transformation proposed by Eggert et al.
            ψtilde =  - sqrt{ψp} D( 1/ψp D(\sqrt{ψp}) ),
            where D is the differential operator w.r.t the varibale t.

1. For S-L problems on a finite domain I=(a,b) with algebraic decay at the endpoints:
DomainSL = FiniteSL.
2. For S-L problems on a infinite domain I=(-∞,∞) with algebraic decay at the endpoints:
DomainSL = Infinite1SL.
3. For S-L problems on a infinite domain I=(-∞,∞) with single-exponential decay at the endpoints:
DomainSL = Infinite2SL.
4. For S-L problems on a semi-infinite domain I=(0,∞) with single-exponential decay at infinity and algebraic decay at 0:
DomainSL = SemiInfiniteSL.
=#

type DomainSL
        ψ::Function
        ψp::Function
        ψtilde::Function
end
FiniteSL(a::Number,b::Number) = DomainSL(t->(b-a)/(1.0+exp(-2t)).+ a , t->(b-a).*sech(t).^2 ./2, ones)
Infinite1SL = DomainSL(sinh, cosh, t->1/4.-3/4 .*sech(t).^2)
Infinite2SL = DomainSL(identity, ones, zeros)
SemiInfiniteSL = DomainSL(t->log1p(exp(t)), t->1./(1+exp(-t)), t-> (1/8).* sech((1/2).*t).^2 .+ 1.0./(2.0.+2exp(t)).^2 )

#=_____________________________________ DEFINITION OF INNER MAP________________________________________
Inner map is given by:
H(t) = u[0]sinh(t) + u[1] + u[2]t + u[3] t^2 + ... + u[n]t^(n-1).
For more information on the form of the inner map, please see reference [3].
The functions Hp(t), Hpp(t) and Hppp(t) correspond to the first, second and third derivative of H(t) respectively.
Input: t    :: Number{T} or Vector{T}
       u[0] :: Number{T}
       u    :: Vector{T}  ( u = [ u[1],u[2],...,u[n] ] )
Ouput: H(t), Hp(t), Hpp(t), or Hppp(t)
=#
function H{T<:Number}(t::T,u0::T,u::Vector{T})
nu = length(u)
u0*sinh(t) + dot(u,t.^[0:nu-1])
end
function H{T<:Number}(t::Vector{T},u0::T,u::Vector{T})
nu = length(u)
u0*sinh(t) .+ t.^([0:nu-1]')*u
end
##################
function Hp{T<:Number}(t::T,u0::T,u::Vector{T})
nu = length(u)
u0*cosh(t) + dot(u[2:nu],[1:nu-1].*t.^[0:nu-2])
end
function Hp{T<:Number}(t::Vector{T},u0::T,u::Vector{T})
nu = length(u)
u0*cosh(t) .+ t.^([0:nu-2]')*([1:nu-1].*u[2:nu])
end
##################
function Hpp{T<:Number}(t::T,u0::T,u::Vector{T})
nu = length(u)
u0*sinh(t) + dot(u[3:nu],[2:nu-1].*[1:nu-2].*t.^[0:nu-3])
end
function Hpp{T<:Number}(t::Vector{T},u0::T,u::Vector{T})
nu = length(u)
u0*sinh(t) .+ t.^([0:nu-3]')*([2:nu-1].*[1:nu-2].*u[3:nu])
end
##################
function Hppp{T<:Number}(t::T,u0::T,u::Vector{T})
nu = length(u)
u0*cosh(t) + dot(u[4:nu],[3:nu-1].*[2:nu-2].*[1:nu-3].*t.^[0:nu-4])
end
function Hppp{T<:Number}(t::Vector{T},u0::T,u::Vector{T})
nu = length(u)
u0*cosh(t) .+ t.^([0:nu-4]')*([3:nu-1].*[2:nu-2].*[1:nu-3].*u[4:nu])
end
##################################################################################################################

################################### The Main function: SincEigen ####################################################
#=
This function computes the eigenvalues of the the Sturm-Liouville (P1) using the double exponential Sinc-Collocation method.
For optimal results, it is worth performing an asymptotic analysis of the solution of P2.
The resulting analysis will lead to the following bounds for constants βL,βR,γL,γR>0:
 |v(x)| < AL exp(-βL exp ( γL |x| ) ) , on (-∞,0)
 |v(x)| < AR exp(-βR exp ( γR |x| ) ) , on (0, ∞)
The parameters βL, βR, γL, γR are used in the calculation of the mesh size, h, for the DESCM.

Another important parameter involved in the calculations of the mesh size for the DESCM is the width, d, of the strip D_{d}.
Let S denote the set of complex singularities of the functions qtilde(x) and ρtilde(x):
S = { z ∈ C : qtilde(z) or ρtilde(z) does not exist a.}
let s denote the positive imaginary part of nearest singularity to the real axis:
s = min | Im{S} |,
then d = min{ π/2max{γL,γR} , s } .

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
_______________________________________________________________________________________________________________________
=#
function SincEigen{T<:Number}(q::Function,ρ::Function,domain::DomainSL,β::Vector{T},γ::Vector{T},dopt::T;enum::Vector{T}=[NaN,NaN], tol::T=5e-12, Range::Vector{Int64}=[1:1:100], u0::T=one(T), u::Vector{T}=[zero(T)], Trace_Mesh::Bool=false)
# Functions used in Matrix construction.
ϕ(t) = domain.ψ(H(t,u0,u))
ϕp2(t) = (domain.ψp(H(t,u0,u)).*Hp(t,u0,u)).^2
qtilde(t) = domain.ψtilde(H(t,u0,u)).*Hp(t,u0,u).^2 .+ (3/4).*(Hpp(t,u0,u)./Hp(t,u0,u)).^2 .-(Hppp(t,u0,u)./2Hp(t,u0,u)) .+ q(ϕ(t)).*ϕp2(t)
rhotilde(t) = ρ(ϕ(t)).*ϕp2(t)
# Determining step sizes and left and right collocation points based on asymptotic information.
if γ[1]>γ[2]
        n = M = Range
        gam = γ[1]
        beta = β[1]
        hoptimal = lambert_W(pi*dopt*gam*n/beta)./(gam*n)
        N = max( ceil( (γ[1]/γ[2]).*n .+ log(β[1]/β[2])./ (γ[2].*hoptimal) ) , 0 )
elseif γ[2]>γ[1]
        n = N = Range
        gam = γ[2]
        beta = β[2]
        hoptimal = lambert_W(pi*dopt*gam*n/beta)./(gam*n)
        M = max( floor( (γ[2]/γ[1]).*n .+ log(β[2]/β[1])./ (γ[1].*hoptimal) ) , 0 )
elseif γ[1] == γ[2] && β[1] > β[2]
        n = M = Range
        gam = γ[1]
        beta = β[1]
        hoptimal = lambert_W(pi*dopt*gam*n/beta)./(gam*n)
        N = ceil( n .+ log(β[1]/β[2])./ (γ[2].*hoptimal) )
elseif γ[1] == γ[2] && β[1] < β[2]
        n = N = Range
        gam = γ[2]
        beta = β[2]
        hoptimal = lambert_W(pi*dopt*gam*n/beta)./(gam*n)
        M = floor(n .+ log(β[2]/β[1])./ (γ[1].*hoptimal) )
elseif γ[1] == γ[2] && β[1] == β[2]
        n = N = Range
        M = 1.0*N
        gam = γ[2]
        beta = β[2]
        if Trace_Mesh == true
        diag_sinc_matrix(t) = (domain.ψtilde(H(t,u0,u) .+ (3/4).*(Hpp(t,u0,u)./Hp(t,u0,u).^2).^2 .-(Hppp(t,u0,u)./2Hp(t,u0,u).^3))./domain.ψp(H(t,u0,u)).^2 .+ q(ϕ(t)))./ρ(ϕ(t))
        hoptimal = [optimize(h->sum( diag_sinc_matrix([-N[i]:N[i]]*h).+ (pi^2/(3h^2))./ rhotilde([-N[i]:N[i]]*h) ),0.001,(3.0*u0+log(pi*dopt*gam*i/beta))./(gam*i)).minimum for i in [1:length(n)]]
        elseif Trace_Mesh == false
        hoptimal = lambert_W(pi*dopt*gam*n/beta)./(gam*n)
        end #if loop
end # if loop

#  INITIAL CONDITIONS
Length = length(Range)                                # Length of the vectors in the following algorithm given the previous conditions
Eigenvalue_enum = zeros(Length)                       # Vector used for storing approximations to eigenvalue enum at every iteration.
MatrixSizes = N.+M.+1                                 # Array of all square matrix sizes for every iteration.
All_Eigenvalues = zeros(int(MatrixSizes[end]),Length) # Matrix used for storing all obtained eigenvalues as columns at every iteration.
for i = 1:Length
h = hoptimal[i]                                     # Mesh size used at every iteration.
k = [-M[i]:N[i]]                                    # Vector of Mesh points.
A = Symmetric(diagm(qtilde(k*h))-Sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
D2 = Symmetric(diagm(rhotilde(k*h)))                  # Construct Diagonal pos. def. matrix D^2
E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
All_Eigenvalues[1:length(E),i] = E                  # Storing all eigenvalues in columns in matrix All_Eigenvalues.
end
## Ouputing the convergence analysis of the algorithm given the number of iterations and the tolerance level tol.
(RESULTS,All_Abs_Error_Approx) = Convergence_Analysis(All_Eigenvalues,tol,MatrixSizes,enum)
(RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes)
end
######################################################################################################################################

################################### Convergence Analysis ####################################################
#=
The following function performs a convergence analysis of the DESCM. More specifically, After running the function
SincEigen for the values "Range", the function Convergence_Analysis will find all eigenvalues which achieved
an approximation to the absolute error less than a prediscribed accuray "tol".
Input: All_Eigenvalues::Matrix{T},         Matrix containing all eigenvalues computed by running from Range[1] to Range[end]
       tol::T,                             Tolerance level
       MatrixSizes::Vector{T},             Vector of matrix size used in the algorithm SincEigen
       enum::Vector{T},                    Vector of given eigenvalue number and corresponding eigenvalue if known.

Output:
RESULTS=[Eig_number MatrixSizeOpt Eigenvalues Error]::Matrix{T},
    Matrix of eigenvalues wich achieved an approximation to the absolute error less than a prediscribed accuray "tol"
    First column: The eigenvalue number,
    Second column: The optimal matrix size needed to achieve "tol".
    Third column: Estimated eigenvalue
    Fourth column: Approximation to the absolute error
All_Abs_Error_Approx::Matrix{T}
    Matrix containing the sequence of approximations to the absolute error for every eigenvalue obtained in the SincEigen algorithm
Row i, Column j = approximations to the absolute error for eigenvalue number "n=i-1" when the matrix size = MatrixSizes[j].
=#

function  Convergence_Analysis{T<:Number}(All_Eigenvalues::Matrix{T},tol::T,MatrixSizes::Vector{T},enum::Vector{T})
# Calculating an Approximation to the Absolute Error for all Energy values
if isnan(enum[1]) || isnan(enum[2]) == true
All_Abs_Error_Approx  = abs(All_Eigenvalues[:,2:end].-All_Eigenvalues[:,1:end-1])
else
All_Abs_Error_Approx = zeros(All_Eigenvalues[:,2:end])
All_Abs_Error_Approx[[1:int(enum[1]),(int(enum[1])+2):end],:] =  abs(All_Eigenvalues[[1:int(enum[1]),(int(enum[1])+2):end],2:end].-All_Eigenvalues[[1:int(enum[1]),(int(enum[1])+2):end],1:end-1])
All_Abs_Error_Approx[int(enum[1])+1,:] =  abs(All_Eigenvalues[int(enum[1])+1,2:end] .- enum[2])
end # if loop

#Finding which Eigenvalues satisfy the condition Absolute Error Approxmiation < tol
m = size(All_Abs_Error_Approx)[1]
First_Non_Zero = [findfirst(All_Abs_Error_Approx[i,:]) for i in [1:m]]
Less_than_tol_matrix =  All_Abs_Error_Approx .< tol
Eig_less_than_tol_posi = [findnext(Less_than_tol_matrix[i,:], true, First_Non_Zero[i]) for i in [1:m]]
# list of eigenvalues  that satisfy the condition Relative Error Approxmiation<tol
Posi_index = findn(Eig_less_than_tol_posi)[1]
# number of eigenvalues that satisfy the condition Error<tol
num_eig_less_than_tol = length(Posi_index)
# Find the optimal value of N for the eignvalues from the list Posi_index
idx = [Eig_less_than_tol_posi[Posi_index[j]]+1 for j in [1:num_eig_less_than_tol]]
# Find the value of the eigenvalues and Relative Error Approximation for the eignvalues from the list Posi_index
Error = [All_Abs_Error_Approx[Posi_index[i],idx[i]-1] for i in [1:num_eig_less_than_tol]]
Eigenvalues = [All_Eigenvalues[Posi_index[i],idx[i]] for i in [1:num_eig_less_than_tol]]
# Display wanted results
Eig_number = Posi_index.-1
MatrixSizeOpt = MatrixSizes[idx]
([Eig_number MatrixSizeOpt Eigenvalues Error], All_Abs_Error_Approx)
end
######################################################################################################################################

################################### The Lambert-W function (See reference [4]) ###################################
#=
The following algorithm computes the Lambert-W function to 15 correct digits using Halley's method for any x
in the domain I=( -1/exp(1), ∞ ). This function can also compute the lambert-W function component-wise for
vectors or matrices. The Lambert-W function is used in the module DESincEig to compute the mesh size h.
Input: x:: Number, Vector or Matrix
Output: Lambert_W(x)
=#
function lambert_W{T<:Number}(x::T)
    if x < -1/exp(1); return NaN; end
    w0 = 1.0
    w1 = w0 - (w0 * exp(w0) - x)/((w0 + 1) * exp(w0) -
        (w0 + 2) * (w0 * exp(w0) - x)/(2 * w0 + 2))
    n = 1
    while abs(w1 - w0) > 1e-15 && n <= 20
        w0 = w1
        w1 = w0 - (w0 * exp(w0) - x)/((w0 + 1) * exp(w0) -
            (w0 + 2) * (w0 * exp(w0) - x)/(2 * w0 + 2))
        n += 1
    end
    return w1
end
@vectorize_1arg Number lambert_W
##################################################################################################################

end #Module
