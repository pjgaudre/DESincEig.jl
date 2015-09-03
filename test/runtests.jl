using DESincEig, SincFun, Base.Test

# Testing the main function SincEigen.

println("Testing Example 1")
f(x) = (35/4)./x.^2  .- 2 .+ x.^2 /16
(RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes) = SincEigen( f , ones , SemiInfinite1{Float64}() , [1.5,0.03125] , [1.0,2.0] , pi/4, tol=1e-10 )
println(RESULTS)
@test norm(RESULTS[1:7,3]-[0.0:6.0]) < sqrt(eps())

println("Testing anharmonic oscillators")
# Exact values for the ground states from E. J. Weniger, Ann. Phys. 246:133--165, 1996.
V(x,β) = x.^2 .+ β.*x.^4
β = [0.2 1.0 4.0]
exact_values = [1.11829265436704,1.39235164153029,1.90313694545900]
for i=1:length(β)
    (RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes) = SincEigen( x->V(x,β[i]) , ones ,Infinite2{Float64}(), [0.125,0.125] , [2.0,2.0] , pi/4 ) # TODO: Are these constants β-dependent?
    @test norm(RESULTS[1,3] - exact_values[i]) < 1e-8
end

#TODO Sextic and Octic, but these probably change more parameters.

V(x,β) = x.^2 .+ β.*x.^6
β = [0.2 1.0 4.0]
exact_values = [1.173889345,1.435624619,1.830437344]

V(x,β) = x.^2 .+ β.*x.^8
β = [0.2 1.0 4.0]
exact_values = [1.24103,1.49102,1.82218]



#TODO add examples of SincEigenStop with converged eigenfunctions.
