using Pkg, BenchmarkTools, Plots
Pkg.add("ForwardDiff")
Pkg.add("ReverseDiff")
using ForwardDiff
using ReverseDiff

import Base: sin, cos, tan


ReLu(x) = x > zero(x) ? x : zero(x)

Rosenbrock(x, y) = (1.0 - x*x) + 100.0*(y - x*x)*(y - x*x)

softmax(arg) = exp.(arg) ./ sum(exp.(arg))


function forwardModeDerivative()
    x = y = collect(-π:0.05:+π)

    # y = x -> ForwardDiff.derivative(ReLu, x)
    # display((y.(x)))
    p1 = plot(x, ForwardDiff.derivative.(sin, x), title = "Sin derivative")
    p2 = plot(x, ForwardDiff.derivative.(cos, x), title = "Cos derivative")
    p3 = plot(x, ForwardDiff.derivative.(tan, x), title = "Tan derivative")
    p4 = plot(x, ForwardDiff.derivative.(ReLu, x), title = "ReLu derivative")
    display(plot(p1, p2, p3, p4, layout = (2, 2), legend = false))
    # y = ForwardDiff.derivative.(softmax, x)
    # display(y)
    # plot(x, dRosenbrockdx.(x, y))

end

function reverseModeDerivative()
    rsin(a) = sum(sin.(a))
    rcos(a) = sum(cos.(a))
    rtan(a) = sum(tan.(a))
    rReLu(a) = sum(ReLu.(a))
    # rsoftmax(a) = sum(softmax.(a))


    x = collect(-π:0.05:+π)

    p1 = plot(x, ReverseDiff.gradient(rsin, x), title = "Sin Derivative")
    p2 = plot(x, ReverseDiff.gradient(rcos, x), title = "Cos Derivative")
    p3 = plot(x, ReverseDiff.gradient(rtan, x), title = "Tan Derivative")
    p4 = plot(x, ReverseDiff.gradient(rReLu, x), title = "ReLu Derivative")
    plot(p1, p2, p3, p4, layout = (2, 2), legend = false)

    # y = ReverseDiff.gradient(rsoftmax, x)
    # display(y)

end


function forwardModeJacobian()
    x = collect(-π/2:0.05:+π)

    fsin(a) = sin.(a)
    fcos(a) = cos.(a)
    ftan(a) = tan.(a)
    fReLu(a) = ReLu.(a)

    display("Jacobi sin")
    display(ForwardDiff.jacobian(fsin, x))
    display("Jacobi cos")
    display(ForwardDiff.jacobian(fcos, x))
    display("Jacobi tan")
    display(ForwardDiff.jacobian(ftan, x))
    display("Jacobi ReLu")
    display(ForwardDiff.jacobian(fReLu, x))
    display("Jacobi softmax")
    display(ForwardDiff.jacobian(softmax, x))
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
    # display("Jacobi softmax")
    # display(ReverseDiff.jacobian(softmax, x))

end



function main()
    # forwardModeDerivative()
    # reverseModeDerivative()
    # reverseModeJacobian()
    # forwardModeJacobian()
end


main()
