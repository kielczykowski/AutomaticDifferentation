include("./FDStructure.jl")
using .FDStructure: Dual, show, value, partials, ReLu, softmax, jacobian, Rosenbrock

import Pkg
Pkg.add("Plots")
using Plots


function main()
    v = collect(0:π/360:π)
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)
    set = collect(0:π/360:2*π)

    p1 = plot(set, partials.([sin(Dual(x,1.0)) for x in set]), title = "Sin derivative")
    p2 = plot(set, partials.([cos(Dual(x,1.0)) for x in set]), title = "Cos derivative")
    p3 = plot(set, partials.([tan(Dual(x,1.0)) for x in set]), title = "Tan derivative")
    p4 = plot(set, partials.([ReLu(Dual(x,1.0)) for x in set]), title = "ReLu derivative")
    pp = plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
    png(pp, "FDBasicPlots")

end

main()
