using DESincEig, SincFun , Winston

println("Testing Example 1")
f(x) = (35/4)./x.^2  .- 2 .+ x.^2 /16
(RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes) = SincEigen( x -> f(x) , ones , SemiInfinite1{Float64}() , [1.5,0.03125] , [1.0,2.0] , pi/4, tol=1e-10 )
println(RESULTS)
p1 = FramedPlot("xlabel","Eigenvalue number","ylabel","Eigenvalues")
Winston.add(p1,Points(RESULTS[:,1],RESULTS[:,3],"type","dot", "color","red"))
setattr(p1, "title", "First few eigenvalues")
display(p1)

println("Testing Example 2")
V(x) = x.^2 .+ x.^4
(RESULTS, All_Abs_Error_Approx , hoptimal , n, MatrixSizes) = SincEigen( x -> V(x) , ones ,Infinite2{Float64}(), [0.125,0.125] , [2.0,2.0] , pi/4 )
println(RESULTS)
p1 = FramedPlot("xlabel","Eigenvalue number","ylabel","Eigenvalues")
Winston.add(p1,Points(RESULTS[:,1],RESULTS[:,3],"type","dot", "color","red"))
setattr(p1, "title", "First few eigenvalues")
display(p1)
