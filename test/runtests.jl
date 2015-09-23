using DESincEig, SincFun, Base.Test

# Testing the main function SincEigen.

println("Testing Example 1")
f(x) = (35/4)./x.^2  .- 2 .+ x.^2 /16
(RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes) = SincEigen( f , ones , SemiInfinite1{Float64}() , [1.5;0.03125] , [1.0;2.0] , pi/4, tol=1e-10 )
@test norm(RESULTS[1:7,3]-collect(0.0:6.0)) < sqrt(eps())

println("Testing quartic anharmonic oscillators")
# Exact values for the ground states from E. J. Weniger, Ann. Phys. 246:133--165, 1996.
V(x,β) = x.^2 .+ β.*x.^4
β = [0.2;1.0;4.0]
exact_values = [1.1182926543696632;1.392351641530291855;1.9031369454587908]

for i=1:length(β)
B = sqrt(β[i])*(1/2)^3 / 3.0
    (RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes) = SincEigen( x->V(x,β[i]) , ones ,Infinite2{Float64}(), [B;B] , [3.0;3.0] , pi/6 ) 
    @test norm(RESULTS[1,3] - exact_values[i]) < 1e-8
end

println("Testing sextic anharmonic oscillators")

# Exact values for the ground state from Chaudhuri1991a and Tater1993, and Flessas1979.
Vsextic(x,c) = c[1]*x.^2 .+ c[2]*x.^4 .+ c[3]*x.^6 
c = [1.0 -4.0 1.0 ; 4.0 -6.0 1.0 ; -7.0 0.0 1.0]
exact_values = [-2.0 ; -9.0 ; -2*sqrt(2) ]
enum = [0 , 1 , 0]
ga = 4.0
B = (1/2)^ga / ga
for i=1:size(c)[2]
    (RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes) = SincEigen( x->V(x,c[i,:]) , ones ,Infinite2{Float64}(), [B;B] , [ga,ga] , pi/2ga ) 
    @test norm(RESULTS[enum+1,3] - exact_values[i]) < 1e-8
end


#TODO add examples of SincEigenStop with converged eigenfunctions.
