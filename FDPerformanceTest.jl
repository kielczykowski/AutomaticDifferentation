include("./FDStructure.jl")
using .FDStructure: Dual, show, value, partials, ReLu, softmax, jacobian, Rosenbrock

import Pkg
Pkg.add("Plots")
using Plots

Pkg.add("BenchmarkTools")
using BenchmarkTools


function main()
    ϵ = Dual(0., 1.)
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

    display("FDPerformance Tests")

    display("Testing sets:")
    display("Small set size: $s_size")
    display("Medium set size: $m_size")
    display("Big set size: $b_size")

    display("Derivatives Calculation")

    display("Sinus small set benchmark")
    display(@benchmark [sin(Dual(x,1.0)) for x in $small_set])
    display("Sinus medium set benchmark")
    display(@benchmark [sin(Dual(x,1.0)) for x in $medium_set])
    display("Sinus big set benchmark")
    display(@benchmark [sin(Dual(x,1.0)) for x in $big_set])

    display("Cos small set benchmark")
    display(@benchmark [cos(Dual(x,1.0)) for x in $small_set])
    display("Cos medium set benchmark")
    display(@benchmark [cos(Dual(x,1.0)) for x in $medium_set])
    display("Cos big set benchmark")
    display(@benchmark [cos(Dual(x,1.0)) for x in $big_set])

    display("Tan small set benchmark")
    display(@benchmark [tan(Dual(x,1.0)) for x in $small_set])
    display("Tan medium set benchmark")
    display(@benchmark [tan(Dual(x,1.0)) for x in $medium_set])
    display("Tan big set benchmark")
    display(@benchmark [tan(Dual(x,1.0)) for x in $big_set])

    display("ReLu small set benchmark")
    display(@benchmark [ReLu(Dual(x,1.0)) for x in $small_set])
    display("ReLu medium set benchmark")
    display(@benchmark [ReLu(Dual(x,1.0)) for x in $medium_set])
    display("ReLu big set benchmark")
    display(@benchmark [ReLu(Dual(x,1.0)) for x in $big_set])

    small_set_negative =  collect(-2*π:π/360:2*π)
    medium_set_negative = collect(-2*π:π/1080:2*π)

    display("Negative Set ReLu")

    display("ReLu small set benchmark")
    display(@benchmark [ReLu(Dual(x,1.0)) for x in $small_set_negative])
    display("ReLu medium set benchmark")
    display(@benchmark [ReLu(Dual(x,1.0)) for x in $medium_set_negative])

    display("Rosenbrock set benchmark")
    display("Rosenbrock dfdx")
    display(@benchmark Rosenbrock.($xv .+ $ϵ, $yv))
    display("Rosenbrock dfdy")
    display(@benchmark Rosenbrock.($xv, $yv .+ $ϵ))

    display("Jacobian Calculation")

    display("Sinus small set benchmark")
    display(@benchmark jacobian(sin, $small_set))
    display("Sinus medium set benchmark")
    display(@benchmark jacobian(sin, $medium_set))
    display("Sinus big set benchmark")
    display(@benchmark jacobian(sin, $big_set))

    display("Cos small set benchmark")
    display(@benchmark jacobian(cos, $small_set))
    display("Cos medium set benchmark")
    display(@benchmark jacobian(cos, $medium_set))
    display("Cos big set benchmark")
    display(@benchmark jacobian(cos, $big_set))

    display("Tan small set benchmark")
    display(@benchmark jacobian(tan, $small_set))
    display("Tan medium set benchmark")
    display(@benchmark jacobian(tan, $medium_set))
    display("Tan big set benchmark")
    display(@benchmark jacobian(tan, $big_set))

    display("ReLu small set benchmark")
    display(@benchmark jacobian(ReLu, $small_set))
    display("ReLu medium set benchmark")
    display(@benchmark jacobian(ReLu, $medium_set))
    display("ReLu big set benchmark")
    display(@benchmark jacobian(ReLu, $big_set))


    display("Rosenbrock set benchmark")
    display(@benchmark jacobian(Rosenbrock, $xv, $yv))

    # display("Softmax using predefined derivatives")
    # display("Softmax small set benchmark")
    # display(@benchmark jacobian(softmax, $small_set))
    # display("Softmax medium set benchmark")
    # display(@benchmark jacobian(softmax, $medium_set))
    # display("Softmax big set benchmark")
    # display(@benchmark jacobian(softmax, $big_set))

end


main()
