cd("C:/Users/Philippe/Documents/Research/Irregular Coulomb Potential")
using SincFun, DESincEig, PyPlot, Distributions
include("C:/Users/Philippe/Documents/Julia/Iregular Potential/Iregular_V1.jl")
Range=collect(1:1:40)
n = 1
@time (RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes, All_Eigenvectors, MN,ap,an) = Laurent_cases(1,Range=Range,tol=5e-10,enum=(n,NaN))

savefig("Potential3_1.eps")

