using Pkg, BenchmarkTools, Plots
Pkg.add("ForwardDiff")
Pkg.add("ReverseDiff")
using ForwardDiff

import Base: sin, cos

function fDiff()
    x = collect(-π:0.05:+π)

    y = x -> ForwardDiff.derivative(cos, x)

    y.(x)
    display((y.(x)))
    plot(x, y.(x))

end

function main()
    fDiff()
end

main()
