function Laurent_cases{S<:Integer,T<:Number}(num::S;tol::T= 5.0e-12,Range::Vector{S}=collect(1:2:55),enum::Tuple{S,Any}=(0,NaN))
srand(13); #123 235
  if 2*Range[1]+1 < enum[1]+1
    Range = collect(enum[1]+1:(Range[2]-Range[1]):Range[end])
end
if         num == 1 # Fernandez PhysicsLettersA 160(1991) 511—514
    ap = [0.0, 1]
    an = [0.0, 0, 0, 0, 0, 9/64]
  #  enum = (0,4.0)
#    enum = (2,NaN)
#    Range=collect(1:1:100)
elseif     num == 2 # Fernandez PhysicsLettersA 160(1991) 511—514
    par1 = 9.0/10
    par2 = 3.0
    par3 = 1.0
    ap = [0, par3^2]
    an = [0, par1*(par1+1)-2*par2*par3, 0, par2*(2*par1-1), 0, par2^2]
    enum = (0, par3*(2*par1+3))
elseif     num == 3 #Varshni
    alpha = 2.0
    gamma2 = 3.0
    beta =-2*sqrt(gamma2)+sqrt(gamma2)*sqrt( 1 + 8*sqrt(alpha*gamma2))
    ap = [0, alpha]
    an = [0, 0, 0, beta, 0, gamma2]
    enum = (0, sqrt(alpha)*(4+beta/sqrt(gamma2)))
elseif     num == 4
    ap = vcat(rand(Uniform(-1, 1),99),[1.0])
    an = vcat(rand(Uniform(-1, 1),99),[1.0])
    #    enum = (0,NaN)
elseif     num == 5
    ap = vcat(rand(Uniform(-5, 5),7),[5.0])
    an = vcat(rand(Uniform(-5, 5),2),[10.0])
#    enum = (0,NaN)
end
# Optimal parameters
gaopt = [(length(an)-2.0)/2 , (length(ap)+2.0)/2]
Bopt = [2*sqrt(an[end])/(length(an)-2) , 2*sqrt(ap[end])/(length(ap)+2)]
dopt = pi/(2*maximum(gaopt))
  #=
q(t) = V(t,ap,an)
β=Bopt
γ=gaopt
tol = 5e-10
=#
(RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes, All_Eigenvectors, MN) = SincEigen_Ireg(t->V(t,ap,an),Bopt,gaopt,dopt,Range=Range,tol=tol,enum=enum)
RESULTS
  PyPlot.clf()
figure(1)
subplot(1,3,1)
title(latexstring("Convergence of DESCM for \$E_{$(enum[1])}\$"))
xlabel(latexstring("Matrix Size: \$M+N+1\$"))
if isnan(enum[2]) == true
ylabel(latexstring("Absolute Error Approximation: \$\\epsilon_{$(enum[1])} \$"))
else
ylabel("Absolute Error")
end
ylim([1e-15,1])
Error = real(All_Abs_Error_Approx[enum[1]+1,:]')
Inthehouse = findnext(Error.<tol, true, 1)+3
xlim([MatrixSizes[2],MatrixSizes[Inthehouse]])
semilogy(MatrixSizes[2:Inthehouse],Error[1:Inthehouse-1],linestyle="--",color="b")

subplot(1,3,3)
Index = findfirst(MatrixSizes,RESULTS[enum[1]+1,2])
k = collect(-MN[Index,1]:1:MN[Index,2])
v_coeff = All_Eigenvectors[1:length(k),Index]
h = hoptimal[Index]
extreme=[1.0]
temp1 = wavefunction(extreme,h,k,v_coeff)
while  abs(temp1[1])>0.0005
   extreme[1]+=1.0
   temp1 =  wavefunction(extreme,h,k,v_coeff)
end
x=linspace(0.00000001,2,300)
#x = linspace(0.00000001,6,300)
W = wavefunction(x,h,k,v_coeff)
title(latexstring("Wavefunction: \$\\psi_{$(enum[1])}(x)\$"))
xlabel(latexstring("\$x\$"))
#ylabel(latexstring("\$\\psi_{$(enum[1])}(x)\$"))
plot(x,W,color="r")

subplot(1,3,2)
Potential = V(convert(Array{Float64,1},x),ap,an)
title(L"Potential Function: $V(x)$")
xlabel(L"x")
#ylabel(L"V(x)")
ylim([minimum(Potential)-20,200]) #29397.608029716015
A=plot(x,Potential,color="k",label=L"V(x)")
B=plot(x,vec(repmat([RESULTS[enum[1]+1,3]],1,length(x))),label=latexstring("\$E_{$(enum[1])}\$"),linestyle="--")
legend()
return (RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes, All_Eigenvectors, MN,ap,an)
end
##################################################################################################
function V{T<:Number}(t::T,ap::Vector{T},an::Vector{T})
  p,n = length(ap),length(an)
  dot(an,1./t.^collect(1:n))+dot(ap,t.^collect(1:p))
end
function V{T<:Number}(t::Vector{T},ap::Vector{T},an::Vector{T})
  p,n = length(ap),length(an)
  (1./t.^(collect(1:n)'))*an .+ (t.^(collect(1:p)'))*ap
end

function SincEigen_Ireg{T<:Number,S<:Integer}(q::Function,β::Vector{T},γ::Vector{T},dopt::T; enum::Tuple{S,Any}=(0,NaN), tol::T=5e-12, Range::Vector{S}=collect(1:1:40))
  qtilde(t) = exp(2t).*q(exp(t)).+0.25
  rhotilde(t) = exp(2t)
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
    hoptimal = lambertW(pi*dopt*gam*n/beta)./(gam*n)
  end # if loop

  #  INITIAL CONDITIONS
  Length = length(Range)                                            # Length of the vectors in the following algorithm given the previous conditions
  MatrixSizes = N.+M.+1                                       # Array of all square matrix sizes for every iteration.
  All_Eigenvalues = zeros(round(Int,MatrixSizes[end]),Length) # Matrix used for storing all obtained eigenvalues as columns at every iteration.
  All_Eigenvectors = zeros(round(Int,MatrixSizes[end]),Length) # Matrix used for storing all obtained eigenvalues as columns at every iteration.
@inbounds for i = 1:Length
    h = hoptimal[i]                                        # Mesh size used at every iteration.
    k = collect(-M[i]:N[i])                                # Vector of Mesh points.
    A = Symmetric(diagm(qtilde(k*h))-sinc(2,k'.-k)/h^2)    # Construct symmetric A matrix.
    D2 = Symmetric(diagm(rhotilde(k*h)))                   # Construct Diagonal pos. def. matrix D^2
    eig_and_vec = eigfact!(A,D2)                                # Solving Generalized eigenvalue problem.
    E = eig_and_vec[:values]
    vec = eig_and_vec[:vectors][:,enum[1]+1]

    All_Eigenvalues[1:length(E),i] = E                     # Storing all eigenvalues in columns in matrix All_Eigenvalues
    All_Eigenvectors[1:length(vec),i] = vec                     # Storing all eigenvalues in columns in matrix All_Eigenvalues.
  end
  isnan(enum[2])
  # Calculating an Approximation to the Absolute Error for all Energy values
  if isnan(enum[2]) == true
    All_Abs_Error_Approx  = abs(All_Eigenvalues[:,2:end].-All_Eigenvalues[:,1:end-1])
  else
    All_Abs_Error_Approx = zeros(All_Eigenvalues[:,2:end])
    Index = [collect(1:enum[1]);collect((enum[1]+2):Length)]
    All_Abs_Error_Approx[Index,:] =  abs(All_Eigenvalues[Index,2:end].-All_Eigenvalues[Index,1:(end-1)])
    All_Abs_Error_Approx[enum[1]+1,:] =  abs(All_Eigenvalues[enum[1]+1,2:end] .- enum[2])
  end # if loop
  ## Ouputing the convergence analysis of the algorithm given the number of iterations and the tolerance level tol.
  RESULTS = Convergence_Analysis(All_Eigenvalues,tol,MatrixSizes,All_Abs_Error_Approx)
  (RESULTS, All_Abs_Error_Approx, hoptimal, n, MatrixSizes,All_Eigenvectors,hcat(M,N))
end

function  Convergence_Analysis{T<:Number}(All_Eigenvalues::Matrix{T},tol::T,MatrixSizes::Vector{T},All_Abs_Error_Approx::Matrix{T})
    #Finding which Eigenvalues satisfy the condition Absolute Error Approxmiation < tol
    m = size(All_Abs_Error_Approx,1)
    First_Non_Zero = [findfirst(All_Abs_Error_Approx[i,:]) for i in collect(1:m)]
    Less_than_tol_matrix =  All_Abs_Error_Approx .< tol
    Eig_less_than_tol_posi = [findnext(Less_than_tol_matrix[i,:], true, First_Non_Zero[i]+1) for i in collect(1:m)]
    # list of eigenvalues  that satisfy the condition Relative Error Approxmiation<tol
    Posi_index = findn(Eig_less_than_tol_posi)#[1]
    # number of eigenvalues that satisfy the condition Error<tol
    num_eig_less_than_tol = length(Posi_index)
    # Find the optimal value of N for the eignvalues from the list Posi_index
    idx = [Eig_less_than_tol_posi[Posi_index[j]]+1 for j in collect(1:num_eig_less_than_tol)]
    # Find the value of the eigenvalues and Relative Error Approximation for the eignvalues from the list Posi_index
    Error = [All_Abs_Error_Approx[Posi_index[i],idx[i]-1] for i in collect(1:num_eig_less_than_tol)]
    Eigenvalues = [All_Eigenvalues[Posi_index[i],idx[i]] for i in collect(1:num_eig_less_than_tol)]
    # Display wanted results
    Eig_number = Posi_index.-1
    MatrixSizeOpt = MatrixSizes[idx]
    [Eig_number MatrixSizeOpt Eigenvalues Error]
end

function wavefunction{T<:Number}(x::LinSpace{T},h::T,k::Vector{T},v=Vector{T})
xk=linspace(0.00000001,100,300000)
wavefunction_squared = (sqrt(xk).*(sinc(log(xk)/h.-k')*v)).^2
Normalizing_Constant = sum((xk[2:end]-xk[1:end-1]).*(wavefunction_squared[2:end]+wavefunction_squared[1:end-1]))./2
#
return sqrt(x).*(sinc(log(x)/h.-k')*v)/sqrt(Normalizing_Constant)
end
function wavefunction{T<:Number}(x::Vector{T},h::T,k::Vector{T},v=Vector{T})
xk=linspace(0.00000001,100,300000)
wavefunction_squared = (sqrt(xk).*(sinc(log(xk)/h.-k')*v)).^2
Normalizing_Constant = sum((xk[2:end]-xk[1:end-1]).*(wavefunction_squared[2:end]+wavefunction_squared[1:end-1]))./2
#
return sqrt(x).*(sinc(log(x)/h.-k')*v)/sqrt(Normalizing_Constant)
end
