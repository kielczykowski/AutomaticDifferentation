using Pkg, BenchmarkTools, Plots
Pkg.add("ForwardDiff")
Pkg.add("ReverseDiff")
using ForwardDiff
using ReverseDiff

import Base: sin, cos, tan

# sin(d::ForwardDiff.Dual{T}) where {T} = ForwardDiff.Dual{T}(sin(value(d)), cos(value(d)) * partials(d))
#
# sin(d::Array{ForwardDiff.Dual}) = sin.(d)

ReLu(x) = x > zero(x) ? x : zero(x)

Rosenbrock(x, y) = (1.0 - x*x) + 100.0*(y - x*x)*(y - x*x)

# f(x::Vector) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);
dRosenbrockdx = (x, y) -> ForwardDiff.derivative(Rosenbrock, x, y)



function forwardModeDerivative()
    x = y = collect(-π:0.05:+π)

    # y = x -> ForwardDiff.derivative(ReLu, x)
    # display((y.(x)))
    p1 = plot(x, ForwardDiff.derivative.(sin, x), title = "Sin derivative")
    p2 = plot(x, ForwardDiff.derivative.(cos, x), title = "Cos derivative")
    p3 = plot(x, ForwardDiff.derivative.(tan, x), title = "Tan derivative")
    p4 = plot(x, ForwardDiff.derivative.(ReLu, x), title = "ReLu derivative")
    plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
    # plot(x, dRosenbrockdx.(x, y))

end

function reverseModeDerivative()
    rsin(a) = sum(sin.(a))
    rcos(a) = sum(cos.(a))
    rtan(a) = sum(tan.(a))
    rReLu(a) = sum(ReLu.(a))

    x = collect(-π:0.05:+π)

    p1 = plot(x, ReverseDiff.gradient(rsin, x), title = "Sin Derivative")
    p2 = plot(x, ReverseDiff.gradient(rcos, x), title = "Cos Derivative")
    p3 = plot(x, ReverseDiff.gradient(rtan, x), title = "Tan Derivative")
    p4 = plot(x, ReverseDiff.gradient(rReLu, x), title = "ReLu Derivative")
    plot(p1, p2, p3, p4, layout = (2, 2), legend = false)

end


function reverseModeJacobian()
    x = collect(-π:0.05:+π)
    rsin(a) = sin.(a)
    rcos(a) = cos.(a)
    rtan(a) = tan.(a)
    rReLu(a) = ReLu.(a)

    display("Jacobi sin")
    display(ReverseDiff.jacobian(rsin, x))
    display("Jacobi cos")
    display(ReverseDiff.jacobian(rcos, x))
    display("Jacobi tan")
    display(ReverseDiff.jacobian(rtan, x))
    display("Jacobi ReLu")
    display(ReverseDiff.jacobian(rReLu, x))

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
    # forwardModeDerivative()
    # reverseModeDerivative()
    reverseModeJacobian()
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
