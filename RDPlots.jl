include("./RDStructure.jl")
using .RDStructure: Node, LeafNode, Variable, ComputableNode, CachedNode,
                   forward, backward, gradient, value, ReLu, softmax,
                   functionValue, functionGradient, Rosenbrock, jacobian
import Pkg
Pkg.add("Plots")
using Plots


function main()
    v = collect(0:π/360:π)
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)
    set = collect(0:π/360:π)
    p1 = plot(set, functionGradient.(sin, set), title = "Sin derivative")
    p2 = plot(set, functionGradient.(cos, set), title = "Cos derivative")
    p3 = plot(set, functionGradient.(tan, set), title = "Tan derivative")
    p4 = plot(set, functionGradient.(ReLu, set), title = "ReLu derivative")
    pp = plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
    png(pp, "RDBasicDerivs")
    display(pp)
    v = -1:0.2:+1
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)

    zv = functionValue.(Rosenbrock, xv, yv)
    dz = functionGradient.(Rosenbrock, xv[:], yv[:])
    contour(v, v, zv, fill=true)
    display(quiver!(xv[:], yv[:], gradient=dz))
    png("RDRosenBrock")

    # p5 = plot(set, functionGradient.(Rosenbrock, xv, yv), title = "Pow derivative")


end

main()
