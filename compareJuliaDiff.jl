using Pkg, BenchmarkTools, Plots
Pkg.add("ForwardDiff")
Pkg.add("ReverseDiff")
using ForwardDiff

import Base: sin, cos, tan

# sin(d::ForwardDiff.Dual{T}) where {T} = ForwardDiff.Dual{T}(sin(value(d)), cos(value(d)) * partials(d))
#
# sin(d::Array{ForwardDiff.Dual}) = sin.(d)

ReLu(x) = x > zero(x) ? x : zero(x)

# f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);


function fDiff()
    x = collect(-π:0.05:+π)

    y = x -> ForwardDiff.derivative(ReLu, x)

    y.(x)
    display((y.(x)))
    plot(x, y.(x))
end


function fDiffJacobian()
    # display("chuj")
    # x= [i for i in -1.0:0.5:1];
    range = [i for i in 0:π/360:2*π] ;
    rangeTan = [i for i in -π/2+π/180:π/180:π/2- π/180];
    x = Base.Vector(-1.0:0.5:1)
    display(x)
    # ForwardDiff.jacobian()

    jacobi = x -> ForwardDiff.derivative(sin, x)

    display("Jacobi Relu")
    display(jacobi(x))
end


function main()
    fDiff()
    # fDiffJacobian()
end

# f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);
#
# x = rand(5)
# display(x)
# #
main()
# g = x -> ForwardDiff.gradient(sin, x)
# #
# display(g(x))
# #
# # display(ForwardDiff.jacobian(f, x))
