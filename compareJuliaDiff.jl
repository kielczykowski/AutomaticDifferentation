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

    v = 0:π/45:2*π
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)
    #
    p1 = plot(v, ReverseDiff.gradient(rsin, v), title = "Sin Derivative")
    p2 = plot(v, ReverseDiff.gradient(rcos, v), title = "Cos Derivative")
    p3 = plot(v, ReverseDiff.gradient(rtan, v), title = "Tan Derivative")
    p4 = plot(v, ReverseDiff.gradient(rReLu, v), title = "ReLu Derivative")
    plot(p1, p2, p3, p4, layout = (2, 2), legend = false)

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

function benchmarkReverseDiffp1()
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

    display("Sin small_set derivative")
    display(@benchmark ReverseDiff.gradient($rsin, $small_set))
    display("Sin medium_set derivative")
    display(@benchmark ReverseDiff.gradient($rsin, $medium_set))
    display("Sin big_set derivative")
    display(@benchmark ReverseDiff.gradient($rsin, $big_set))

    display("Cos small_set derivative")
    display(@benchmark ReverseDiff.gradient($rcos, $small_set))
    display("cos medium_set derivative")
    display(@benchmark ReverseDiff.gradient($rcos, $medium_set))
    display("cos big_set derivative")
    display(@benchmark ReverseDiff.gradient($rcos, $big_set))

    display("tan small_set derivative")
    display(@benchmark ReverseDiff.gradient($rtan, $small_set))
    display("tan medium_set derivative")
    display(@benchmark ReverseDiff.gradient($rtan, $medium_set))
    display("tan big_set derivative")
    display(@benchmark ReverseDiff.gradient($rtan, $big_set))

    display("ReLu small_set derivative")
    display(@benchmark ReverseDiff.gradient($rReLu, $small_set))
    display("ReLu medium_set derivative")
    display(@benchmark ReverseDiff.gradient($rReLu, $medium_set))
    display("ReLu big_set derivative")
    display(@benchmark ReverseDiff.gradient($rReLu, $big_set))

    display("Rosenbrock derivative")
    display(@benchmark ReverseDiff.gradient($rRosenbrock, ($xv, $yv)))

end


function benchmarkReverseDiffp2()
    display("Jacobian")
    rsin(a) = sin.(a)
    rcos(a) = cos.(a)
    rtan(a) = tan.(a)
    rReLu(a) = ReLu.(a)
    rRosenbrock(a, b) = Rosenbrock.(a, b)

    display("Jacobi small_set sin")
    display(@benchmark ReverseDiff.jacobian($rsin, $small_set))
    display("Jacobi medium_set sin")
    display(@benchmark ReverseDiff.jacobian($rsin, $medium_set))
    display("Jacobi big_set sin")
    display(@benchmark ReverseDiff.jacobian($rsin, $big_set))

    display("Jacobi small_set cos")
    display(@benchmark ReverseDiff.jacobian($rcos, $small_set))
    display("Jacobi medium_set cos")
    display(@benchmark ReverseDiff.jacobian($rcos, $medium_set))
    display("Jacobi big_set cos")
    display(@benchmark ReverseDiff.jacobian($rcos, $big_set))

    display("Jacobi small_set tan")
    display(@benchmark ReverseDiff.jacobian($rtan, $small_set))
    display("Jacobi medium_set tan")
    display(@benchmark ReverseDiff.jacobian($rtan, $medium_set))
    display("Jacobi big_set tan")
    display(@benchmark ReverseDiff.jacobian($rtan, $big_set))

    display("Jacobi small_set ReLu")
    display(@benchmark ReverseDiff.jacobian($rReLu, $small_set))
    display("Jacobi medium_set ReLu")
    display(@benchmark ReverseDiff.jacobian($rReLu, $medium_set))
    display("Jacobi big_set ReLu")
    display(@benchmark ReverseDiff.jacobian($rReLu, $big_set))

    display("Jacobi Rosenbrock")
    display(@benchmark ReverseDiff.jacobian($rRosenbrock, $xv, $yv))
end


function benchmarkForwardDiff()
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

    display("ForwardDiff")

    display("Derivative")

    display("Sin small_set deriv")
    display(@benchmark ForwardDiff.derivative.(sin, $small_set))
    display("Sin medium_set deriv")
    display(@benchmark ForwardDiff.derivative.(sin, $medium_set))
    display("Sin big_set deriv")
    display(@benchmark ForwardDiff.derivative.(sin, $big_set))

    display("Cos small_set deriv")
    display(@benchmark ForwardDiff.derivative.(cos, $small_set))
    display("Cos medium_set deriv")
    display(@benchmark ForwardDiff.derivative.(cos, $medium_set))
    display("Cos big_set deriv")
    display(@benchmark ForwardDiff.derivative.(cos, $big_set))

    display("Tan small_set deriv")
    display(@benchmark ForwardDiff.derivative.(tan, $small_set))
    display("Tan medium_set deriv")
    display(@benchmark ForwardDiff.derivative.(tan, $medium_set))
    display("Tan big_set deriv")
    display(@benchmark ForwardDiff.derivative.(tan, $big_set))

    display("ReLu small_set deriv")
    display(@benchmark ForwardDiff.derivative.(ReLu, $small_set))
    display("ReLu medium_set deriv")
    display(@benchmark ForwardDiff.derivative.(ReLu, $medium_set))
    display("ReLu big_set deriv")
    display(@benchmark ForwardDiff.derivative.(ReLu, $big_set))

    ### display("Rosenbrock deriv")
    ### display(@benchmark ForwardDiff.gradient.(Rosenbrock, ($xv, $yv)))

    display("Jacobian")

    fsin(a) = sin.(a)
    fcos(a) = cos.(a)
    ftan(a) = tan.(a)
    fReLu(a) = ReLu.(a)
    fRosenbrock(a, b) = Rosenbrock.(a, b)

    display("Jacobi small_set sin")
    display(@benchmark ForwardDiff.jacobian($fsin, $small_set))
    display("Jacobi medium_set sin")
    display(@benchmark ForwardDiff.jacobian($fsin, $medium_set))
    display("Jacobi big_set sin")
    display(@benchmark ForwardDiff.jacobian($fsin, $big_set))

    display("Jacobi small_set cos")
    display(@benchmark ForwardDiff.jacobian($fcos, $small_set))
    display("Jacobi medium_set cos")
    display(@benchmark ForwardDiff.jacobian($fcos, $medium_set))
    display("Jacobi big_set cos")
    display(@benchmark ForwardDiff.jacobian($fcos, $big_set))

    display("Jacobi small_set tan")
    display(@benchmark ForwardDiff.jacobian($ftan, $small_set))
    display("Jacobi medium_set tan")
    display(@benchmark ForwardDiff.jacobian($ftan, $medium_set))
    display("Jacobi big_set tan")
    display(@benchmark ForwardDiff.jacobian($ftan, $big_set))

    display("Jacobi small_set ReLu")
    display(@benchmark ForwardDiff.jacobian($fReLu, $small_set))
    display("Jacobi medium_set ReLu")
    display(@benchmark ForwardDiff.jacobian($fReLu, $medium_set))
    display("Jacobi big_set ReLu")
    display(@benchmark ForwardDiff.jacobian($fReLu, $big_set))

    display("Jacobi Rosenbrock")
    display(@benchmark ForwardDiff.jacobian($fRosenbrock, $xv, $yv))

    display("Jacobi small_set softmax")
    display(@benchmark ForwardDiff.jacobian(softmax, $small_set))
    display("Jacobi medium_set softmax")
    display(@benchmark ForwardDiff.jacobian(softmax, $medium_set))
    display("Jacobi big_set softmax")
    display(@benchmark ForwardDiff.jacobian(softmax, $big_set))

end


function main()
    benchmarkReverseDiffp1()
    benchmarkReverseDiffp2()
    # benchmarkForwardDiff()
    # forwardModeDerivative()
    reverseModeDerivative()
    # reverseModeJacobian()
    # forwardModeJacobian()
end


main()
