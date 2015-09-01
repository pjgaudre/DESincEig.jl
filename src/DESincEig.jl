module DESincEig
#=
A module by Philippe Gaudreau,
Department of Mathematical & Statistical Sciences,
University of Alberta, 2014.

The primary function of this module (function SincEigen) computes the eigenvalues of
Singular Sturm-Liouville problems using the DESINC Method.
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

using SincFun, Optim

export SincEigen

include("lambertW.jl")

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
The SincFun type Domain is used to select from the outer maps depending on the domain of the problem.
Concrete Domains have methods:
 1. ψ  : The outer map
 2. ψp : The derivative of ψ, (ψ prime)
 3. ψtilde : The first half of the resulting transformation proposed by Eggert et al.
            ψtilde =  - sqrt{ψp} D( 1/ψp D(\sqrt{ψp}) ),
            where D is the differential operator w.r.t the varibale t.

1. For S-L problems on a finite domain I=(a,b) with algebraic decay at the endpoints:
Domain = Finite(a,b).
2. For S-L problems on a infinite domain I=(-∞,∞) with algebraic decay at the endpoints:
Domain = Infinite1{Float64}().
3. For S-L problems on a infinite domain I=(-∞,∞) with single-exponential decay at the endpoints:
Domain = Infinite2{Float64}().
4. For S-L problems on a semi-infinite domain I=(0,∞) with single-exponential decay at infinity and algebraic decay at 0:
Domain = SemiInfinite1{Float64}().
=#

export ψtilde
ψtilde(d::Finite,t) = 0*t+1
ψtilde(d::Infinite1,t) = 0.25.-0.75.*sech(t).^2
ψtilde(d::Infinite2,t) = 0*t
ψtilde(d::SemiInfinite1,t) = 0.125.* sech(t/2).^2 .+ 0.25./(1.0.+exp(t)).^2
ψtilde(d::SemiInfinite2,t) = 0*t + 0.25

#=_____________________________________ DEFINITION OF INNER MAP________________________________________
Inner map is given by a SincFun type ConformalMap.
It stores the data for h(t) = u₀sinh(t) + u₁ + u₂t + u₃ t^2 + ⋯ + uₙt^(n-1).
For more information on the form of the inner map, please see reference [3].
=#

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
3. domain:: Domain,      Finite, Infinite1, Infinite2 or SemiInfinite1
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
resulting from the DESCM. For even functions q(x) and ρ(x) and an infinite domain: Domain = Infinite1 or Infinite2,
once can minimize this functional to obtain an alternate mesh size: htilde. This alternate mesh-size has proven
to be better suited for highly-oscillatory functions q(x). This functional is minimized using the Julia package: Optim.
_______________________________________________________________________________________________________________________
=#
function SincEigen{T<:Number}(q::Function,ρ::Function,domain::Domain{T},β::Vector{T},γ::Vector{T},dopt::T;enum::Vector{T}=[NaN,NaN], tol::T=5e-12, Range::Vector{Int64}=[1:1:100], u0::T=one(T), u::Vector{T}=[zero(T)], Trace_Mesh::Bool=false)
    # Functions used in Matrix construction.
    H = [ConformalMap(u0,u)]
    for i=1:3 push!(H,H[end]') end
    ϕ(t) = ψ(domain,H[1][t])
    ϕp2(t) = (ψp(domain,H[1][t]).*H[2][t]).^2
    qtilde(t) = ψtilde(domain,H[1][t]).*H[2][t].^2 .+ (3/4).*(H[3][t]./H[2][t]).^2 .-(H[4][t]./2H[2][t]) .+ q(ϕ(t)).*ϕp2(t)
    rhotilde(t) = ρ(ϕ(t)).*ϕp2(t)
    # Determining step sizes and left and right collocation points based on asymptotic information.
    if γ[1]>γ[2]
        n = M = Range
        gam = γ[1]
        beta = β[1]
        hoptimal = lambertW(pi*dopt*gam*n/beta)./(gam*n)
        N = max( ceil( (γ[1]/γ[2]).*n .+ log(β[1]/β[2])./ (γ[2].*hoptimal) ) , 0 )
    elseif γ[2]>γ[1]
        n = N = Range
        gam = γ[2]
        beta = β[2]
        hoptimal = lambertW(pi*dopt*gam*n/beta)./(gam*n)
        M = max( floor( (γ[2]/γ[1]).*n .+ log(β[2]/β[1])./ (γ[1].*hoptimal) ) , 0 )
    elseif γ[1] == γ[2] && β[1] > β[2]
        n = M = Range
        gam = γ[1]
        beta = β[1]
        hoptimal = lambertW(pi*dopt*gam*n/beta)./(gam*n)
        N = ceil( n .+ log(β[1]/β[2])./ (γ[2].*hoptimal) )
    elseif γ[1] == γ[2] && β[1] < β[2]
        n = N = Range
        gam = γ[2]
        beta = β[2]
        hoptimal = lambertW(pi*dopt*gam*n/beta)./(gam*n)
        M = floor(n .+ log(β[2]/β[1])./ (γ[1].*hoptimal) )
    elseif γ[1] == γ[2] && β[1] == β[2]
        n = N = Range
        M = 1.0*N
        gam = γ[2]
        beta = β[2]
        if Trace_Mesh == true
            diag_sinc_matrix(t) = (ψtilde(domain,H[1][t] .+ (3/4).*(H[3][t]./H[2][t].^2).^2 .-(H[4][t]./2H[2][t].^3))./ψp(domain,H[1][t]).^2 .+ q(ϕ(t)))./ρ(ϕ(t))
            hoptimal = [optimize(h->sum( diag_sinc_matrix([-N[i]:N[i]]*h).+ (pi^2/(3h^2))./ rhotilde([-N[i]:N[i]]*h) ),0.001,(3.0*u0+log(pi*dopt*gam*i/beta))./(gam*i)).minimum for i in [1:length(n)]]
        elseif Trace_Mesh == false
            hoptimal = lambertW(pi*dopt*gam*n/beta)./(gam*n)
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
        A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
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


function SincEigenStop{T<:Number}(q::Function , ρ::Function , domain::Domain{T} , β::Vector{T} ,γ::Vector{T} ,dopt::T; enum::Vector{T}=[zero,NaN],tol::T = convert(T,5.0e-12),u0::T = one(T),u::Vector{T}=[zero(T)])
# Functions used in Matrix construction.
    H = [ConformalMap(u0,u)]
    for i=1:3 push!(H,H[end]') end
    ϕ(t) = ψ(domain,H[1][t])
    ϕp2(t) = (ψp(domain,H[1][t]).*H[2][t]).^2
    qtilde(t) = ψtilde(domain,H[1][t]).*H[2][t].^2 .+ (3/4).*(H[3][t]./H[2][t]).^2 .-(H[4][t]./2H[2][t]) .+ q(ϕ(t)).*ϕp2(t)
    rhotilde(t) = ρ(ϕ(t)).*ϕp2(t)

# Determining step sizes and left and right collocation points based on asymptotic information.
if γ[1]>γ[2]

    M = max( floor((enum[1]+1)/2) , 1 )
    h = lambertW(pi*dopt*γ[1]*M/β[1])./(γ[1]*M)
    N = max( ceil( (γ[1]/γ[2]).*M .+ log(β[1]/β[2])./ (γ[2].*h) ) , 0 )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_old = E[enum[1]+1]

    M += 1
    h = lambertW(pi*dopt*γ[1]*M/β[1])./(γ[1]*M)
    N = max( ceil( (γ[1]/γ[2]).*M .+ log(β[1]/β[2])./ (γ[2].*h) ) , 0 )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_new = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])

    while Abs_Error_enum > tol && M < 200
    Eigenvalue_enum_old = Eigenvalue_enum_new
    M += 1
    h = lambertW(pi*dopt*γ[1]*M/β[1])./(γ[1]*M)
    N = max( ceil( (γ[1]/γ[2]).*M .+ log(β[1]/β[2])./ (γ[2].*h) ) , 0 )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_new = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])
    end #while loop

elseif γ[2]>γ[1]

    N = max( floor((enum[1]+1)/2) , 1 )
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    M = max( floor( (γ[2]/γ[1]).*N .+ log(β[2]/β[1])./ (γ[1].*h) ) , 0 )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_old = E[enum[1]+1]

    N += 1
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    M = max( floor( (γ[2]/γ[1]).*N .+ log(β[2]/β[1])./ (γ[1].*h) ) , 0 )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_new = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])

    while Abs_Error_enum > tol && N < 200
    Eigenvalue_enum_old = Eigenvalue_enum_new
    N += 1
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    M = max( floor( (γ[2]/γ[1]).*N .+ log(β[2]/β[1])./ (γ[1].*h) ) , 0 )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_old = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])
    end #while loop

elseif γ[1] == γ[2] && β[1] > β[2]

    M = max( floor((enum[1]+1)/2) , 1 )
    h = lambertW(pi*dopt*γ[1]*M/β[1])./(γ[1]*M)
    N = ceil( (γ[1]/γ[2]).*M .+ log(β[1]/β[2])./ (γ[2].*h) )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_old = E[enum[1]+1]

    M += 1
    h = lambertW(pi*dopt*γ[1]*M/β[1])./(γ[1]*M)
    N = ceil( (γ[1]/γ[2]).*M .+ log(β[1]/β[2])./ (γ[2].*h) )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_new = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])

    while Abs_Error_enum > tol && M < 200
    Eigenvalue_enum_old = Eigenvalue_enum_new
    M += 1
    h = lambertW(pi*dopt*γ[1]*M/β[1])./(γ[1]*M)
    N = ceil( (γ[1]/γ[2]).*M .+ log(β[1]/β[2])./ (γ[2].*h) )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_new = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])
    end #while loop

elseif γ[1] == γ[2] && β[1] < β[2]

    N = max( floor((enum[1]+1)/2) , 1 )
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    M = floor( (γ[2]/γ[1]).*N .+ log(β[2]/β[1])./ (γ[1].*h) )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_old = E[enum[1]+1]

    N += 1
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    M = floor( (γ[2]/γ[1]).*N .+ log(β[2]/β[1])./ (γ[1].*h) )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_new = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])

    while Abs_Error_enum > tol && N < 200
    Eigenvalue_enum_old = Eigenvalue_enum_new
    N += 1
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    M = floor( (γ[2]/γ[1]).*N .+ log(β[2]/β[1])./ (γ[1].*h) )
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_old = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])
    end #while loop

elseif γ[1] == γ[2] && β[1] == β[2]

    N = max( floor((enum[1]+1)/2) , 1 )
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    k = [-N:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_old = E[enum[1]+1]

    N += 1
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    k = [-N:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_new = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])

    while Abs_Error_enum > tol && N < 200
    Eigenvalue_enum_old = Eigenvalue_enum_new
    N += 1
    M = N
    h = lambertW(pi*dopt*γ[2]*N/β[2])./(γ[2]*N)
    k = [-M:N]
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2) # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                # Construct Diagonal pos. def. matrix D^2.
    E = eigvals(A,D2)                                   # Solving Generalized eigenvalue problem.
    Eigenvalue_enum_new = E[enum[1]+1]
    Abs_Error_enum = isnan(enum[2]) ? abs(Eigenvalue_enum_new- Eigenvalue_enum_old) : abs(Eigenvalue_enum_new - enum[2])
    end #while loop
end #if loop

    v = eigvecs(A,D2)[:,enum[1]+1]
    return (enum[1],int(N+M+1),Eigenvalue_enum_new,Abs_Error_enum,v,h,k)

end #function



end #Module
