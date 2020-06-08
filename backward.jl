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


function fgrad(f, xv, yv)
    x, y = Variable(xv), Variable(yv)
    z = f(x, y)
    backward(z, Variable(1.0))
    5e-4x.grad, 5e-4y.grad
end


function testRosenBrock()
    v = -1:.2:+1
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)

    zv = fval.(rosenbrock, xv, yv)
    dz = fgrad.(rosenbrock, xv[:], yv[:])

    zv = reshape(zv, n, n)
    contour(v, v, zv, fill=true)
    display(quiver!(xv[:], yv[:], gradient=dz))
end

function main()
    testRosenBrock()
end

main()
