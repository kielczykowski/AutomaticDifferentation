include("./RDStructure.jl")
using .RDStructure: Node, LeafNode, Variable, ComputableNode, CachedNode,
                   forward, backward, gradient, value
using Plots


rosenbrock(x, y) = (Variable(1.0) - x*x) + Variable(100.0)*(y - x*x)*(y - x*x)


function fval(f, xv, yv)
    x, y = Variable(xv), Variable(yv)
    z = f(x, y)
    value(z)
end


function fval(f, x)
    arg = Variable(x)
    z = f(arg)
    value(z)
end



function fgrad(f, xv, yv)
    x, y = Variable(xv), Variable(yv)
    z = f(x, y)
    backward(z, Variable(1.0))
    5e-4x.grad, 5e-4y.grad
end


function fgrad(f, x)
    arg = Variable(x)
    z = f(arg)
    backward(z, Variable(1.0))
    # display(arg.grad)
    arg.grad
end


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


function main()
    # testRosenBrock()
    # testSin()
    # testCos()
    testTan()
end


main()
