include("./RDStructure.jl")
using .RDStructure: Node, LeafNode, Variable, ComputableNode, CachedNode,
                   forward, backward, gradient, value, ReLu, fval, fgrad
using Plots


rosenbrock(x, y) = (Variable(1.0) - x*x) + Variable(100.0)*(y - x*x)*(y - x*x)


# ReLu(x) = x > zero(x) ? x : zero(x)





function testRosenBrock()
    v = -1:0.2:+1
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)

    zv = fval.(rosenbrock, xv, yv)
    dz = fgrad.(rosenbrock, xv[:], yv[:])

    zv = reshape(zv, n, n)
    contour(v, v, zv, fill=true)
    display(quiver!(xv[:], yv[:], gradient=dz))
end


function testSin()
    range =  0:π/360:2*π
    zv = fval.(sin, range)
    dz = fgrad.(sin, range)

    # display(dz)
    plot(range, zv, label="values")
    display(plot!(range, dz, label = "derivative"))
end


function testCos()
    range =  0:π/360:2*π
    zv = fval.(cos, range)
    dz = fgrad.(cos, range)

    # display(zv)
    plot(range, zv, label="values")
    display(plot!(range, dz, label = "derivative"))
end


function testTan()
    range =  -π/2+0.1:π/360:π/2 - 0.1
    zv = fval.(tan, range)
    dz = fgrad.(tan, range)

    display(zv)
    plot(range, zv, label="values")
    display(plot!(range, dz, label = "derivative"))
end


function testReLu()
    x = -1.0:0.05:+1.0
    # display(ReLu(Variable(-1.0)))
    # display(ReLu.(V)
    zv = fval.(ReLu, x)
    dz = fgrad.(ReLu, x)

    display(zv)
    display(dz)
    plot(x, zv, label="values")
    display(plot!(x, dz, label="derivative"))

end


function main()
    # testRosenBrock()
    # testSin()
    # testCos()
    # testTan()
    testReLu()
end


main()
