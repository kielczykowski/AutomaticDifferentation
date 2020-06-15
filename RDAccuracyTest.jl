include("./RDStructure.jl")
include("./ReferenceDerivatives.jl")
using .RDStructure: Node, LeafNode, Variable, ComputableNode, CachedNode,
                   forward, backward, gradient, value, ReLu, softmax,
                   functionValue, functionGradient, Rosenbrock, jacobian

using .ReferenceDerivatives: dsindx, dcosdx, dtandx, dReLudx, dSoftmaxdx,
                            dRosenbrockdx, dRosenbrockdy

import Pkg
Pkg.add("Plots")
using Plots

Pkg.add("BenchmarkTools")
using BenchmarkTools

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
    measureAccuracy(functionGradient.(sin, test_set), dsindx.(test_set))
    display("Cosinus")
    measureAccuracy(functionGradient.(cos, test_set), dcosdx.(test_set))
    display("Tangent")
    measureAccuracy(functionGradient.(tan, test_set), dtandx.(test_set))
    # display("Max Float64 value")
    # display(floatmax(Float64))
    plot(test_set, functionGradient.(tan, test_set), label = "ReverseAD")
    plot!(test_set, dtandx.(test_set), label = "Mathematical Derivative")
    png("RDAccuracyTangent")
    display("ReLu")
    measureAccuracy(functionGradient.(ReLu, test_set), dReLudx.(test_set))

    display("RosenBrock")
    d = functionGradient.(Rosenbrock, xv, yv)
    dx = [x[1] for x in d]
    dy = [x[2] for x in d]
    display("Rosenbrock dx")
    measureAccuracy(dx, 5e-4dRosenbrockdx.(xv, yv))
    display("Rosenbrock dy")
    measureAccuracy(dy, 5e-4dRosenbrockdy.(xv, yv))



    # dRosenbrockdx.(xv, yv)
    # display(dy)
    # measuerAccuracy()
end


main()
