include("./FDStructure.jl")
include("./ReferenceDerivatives.jl")
using .FDStructure: Dual, show, value, partials, ReLu, softmax, jacobian, Rosenbrock

using .ReferenceDerivatives: dsindx, dcosdx, dtandx, dReLudx, dSoftmaxdx


using Statistics

import Pkg
Pkg.add("Plots")
using Plots



function measureAccuracy(x, x̂)
    difference = x .- x̂
    mn = mean(difference)
    medi = median(difference)
    maximum, _ = findmax(difference)
    display("Mean diffence")
    display("$mn")
    display("Median difference")
    display("$medi")
    display("Max difference")
    display("$maximum")
end


function main()
    test_set = collect(0:π/1080:2*π)
    v = 0:π/45:2*π
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)

    t_size = length(test_set)

    display("RDAccuracy Test")

    display("Testing set:")
    display("Set size: $t_size")

    display("Derivatives")
    display("Sinus")
    measureAccuracy(partials.([sin(Dual(x,1.0)) for x in test_set]), dsindx.(test_set))
    display("Cosinus")
    measureAccuracy(partials.([cos(Dual(x,1.0)) for x in test_set]), dcosdx.(test_set))
    display("Tangent")
    measureAccuracy(partials.([tan(Dual(x,1.0)) for x in test_set]), dtandx.(test_set))
    # display("Max Float64 value")
    # display(floatmax(Float64))
    plot(test_set, partials.([tan(Dual(x,1.0)) for x in test_set]), label = "ForwardAD")
    plot!(test_set, dtandx.(test_set), label = "Mathematical Derivative")
    png("FDAccuracyTangent")
    display("ReLu")
    measureAccuracy(partials.([ReLu(Dual(x,1.0)) for x in test_set]), dReLudx.(test_set))
end


main()
