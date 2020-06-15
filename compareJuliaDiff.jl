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
    # rsin(a) = sum(sin.(a))
    # rcos(a) = sum(cos.(a))
    # rtan(a) = sum(tan.(a))
    # rReLu(a) = sum(ReLu.(a))
    # # rsoftmax(a) = sum(softmax.(a))
    #
    #
    v = 0:π/45:2*π
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)
    #
    # p1 = plot(x, ReverseDiff.gradient(rsin, x), title = "Sin Derivative")
    # p2 = plot(x, ReverseDiff.gradient(rcos, x), title = "Cos Derivative")
    # p3 = plot(x, ReverseDiff.gradient(rtan, x), title = "Tan Derivative")
    # p4 = plot(x, ReverseDiff.gradient(rReLu, x), title = "ReLu Derivative")
    # plot(p1, p2, p3, p4, layout = (2, 2), legend = false)

    rRosenbrock(x, y) = sum(Rosenbrock.(x, y))
    y = ReverseDiff.gradient(rRosenbrock, (xv, yv))
    display(y)

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

function benchmarkReverseDiff()
    small_set =  collect(0:π/360:2*π)
    medium_set = collect(0:π/1080:2*π)
    big_set = collect(-2 * π:π/3600:2*π)

    s_size = length(small_set)
    m_size = length(medium_set)
    b_size = length(big_set)

    # Rosenbrock set
    v = 0:π/45:2*π
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)

    display("ReverseDiff Tests")

    display("Testing sets:")
    display("Small set size: $s_size")
    display("Medium set size: $m_size")
    display("Big set size: $b_size")

    display("Derivative")
    rsin(a) = sum(sin.(a))
    rcos(a) = sum(cos.(a))
    rtan(a) = sum(tan.(a))
    rReLu(a) = sum(ReLu.(a))
    rRosenbrock(a, b) = sum(Rosenbrock.(a, b))

    # display("Sin small_set derivative")
    # display(@benchmark ReverseDiff.gradient($rsin, $small_set))
    # display("Sin medium_set derivative")
    # display(@benchmark ReverseDiff.gradient($rsin, $medium_set))
    # display("Sin big_set derivative")
    # display(@benchmark ReverseDiff.gradient($rsin, $big_set))
    #
    # display("Cos small_set derivative")
    # display(@benchmark ReverseDiff.gradient($rcos, $small_set))
    # display("cos medium_set derivative")
    # display(@benchmark ReverseDiff.gradient($rcos, $medium_set))
    # display("cos big_set derivative")
    # display(@benchmark ReverseDiff.gradient($rcos, $big_set))
    #
    # display("tan small_set derivative")
    # display(@benchmark ReverseDiff.gradient($rtan, $small_set))
    # display("tan medium_set derivative")
    # display(@benchmark ReverseDiff.gradient($rtan, $medium_set))
    # display("tan big_set derivative")
    # display(@benchmark ReverseDiff.gradient($rtan, $big_set))

    display("ReLu small_set derivative")
    display(@benchmark ReverseDiff.gradient($rReLu, $small_set))
    display("ReLu medium_set derivative")
    display(@benchmark ReverseDiff.gradient($rReLu, $medium_set))
    display("ReLu big_set derivative")
    display(@benchmark ReverseDiff.gradient($rReLu, $big_set))

    display("Rosenbrock derivative")
    display(@benchmark ReverseDiff.gradient($rRosenbrock, ($xv, $yv)))
    # p2 = plot(x, ReverseDiff.gradient(rcos, x), title = "Cos Derivative")
    # p3 = plot(x, ReverseDiff.gradient(rtan, x), title = "Tan Derivative")
    # p4 = plot(x, ReverseDiff.gradient(rReLu, x), title = "ReLu Derivative")
end



function main()
    benchmarkReverseDiff()
    # forwardModeDerivative()
    # reverseModeDerivative()
    # reverseModeJacobian()
    # forwardModeJacobian()
end


main()
