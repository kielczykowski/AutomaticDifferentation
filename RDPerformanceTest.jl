include("./RDStructure.jl")
using .RDStructure: Node, LeafNode, Variable, ComputableNode, CachedNode,
                   forward, backward, gradient, value, ReLu, softmax,
                   functionValue, functionGradient, Rosenbrock, jacobian
import Pkg
Pkg.add("Plots")
using Plots

Pkg.add("BenchmarkTools")
using BenchmarkTools

function testOnSet(f::Function, set::Vector)
    @benchmark functionGradient.(f, $set)
end


function main()
    small_set =  collect(0:π/360:2*π)
    medium_set = collect(0:π/1080:2*π)
    big_set = collect(-2 * π:π/3600:2*π)

    display("Derivatives")

    display("Sinus small set benchmark")
    display(@benchmark functionGradient.(sin, $small_set))
    display("Sinus medium set benchmark")
    display(@benchmark functionGradient.(sin, $medium_set))
    display("Sinus big set benchmark")
    display(@benchmark functionGradient.(sin, $big_set))

    display("Cosinus small set benchmark")
    display(@benchmark functionGradient.(cos, $small_set))
    display("Cosinus medium set benchmark")
    display(@benchmark functionGradient.(cos, $medium_set))
    display("Cosinus big set benchmark")
    display(@benchmark functionGradient.(cos, $big_set))

    display("Tan small set benchmark")
    display(@benchmark functionGradient.(tan, $small_set))
    display("Tan medium set benchmark")
    display(@benchmark functionGradient.(tan, $medium_set))
    display("Tan big set benchmark")
    display(@benchmark functionGradient.(tan, $big_set))



    display("ReLu small set benchmark")
    display(@benchmark functionGradient.(ReLu, $small_set))
    display("ReLu medium set benchmark")
    display(@benchmark functionGradient.(ReLu, $medium_set))
    display("ReLu big set benchmark")
    display(@benchmark functionGradient.(ReLu, $big_set))

    # negative value in datasets make algorithm work significantly longer
    small_set_negative =  collect(-2*π:π/360:2*π)
    medium_set_negative = collect(-2*π:π/1080:2*π)

    display("ReLu small set benchmark")
    display(@benchmark functionGradient.(ReLu, $small_set_negative))
    display("ReLu medium set benchmark")
    display(@benchmark functionGradient.(ReLu, $medium_set_negative))

    display("Jacobian")

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

    display("Softmax small set benchmark")
    display(@benchmark jacobian(softmax, $small_set))
    display("Softmax medium set benchmark")
    display(@benchmark jacobian(softmax, $medium_set))
    display("Softmax big set benchmark")
    display(@benchmark jacobian(softmax, $big_set))










end


main()
