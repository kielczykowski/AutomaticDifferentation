include("./RDStructure.jl")
using .RDStructure: Node, LeafNode, Variable, ComputableNode, CachedNode,
                   forward, backward, gradient, value, ReLu,
                   functionValue, functionGradient, Rosenbrock, jacobian
using Plots





function testRosenBrock()
    v = -1:0.2:+1
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)

    zv = functionValue.(Rosenbrock, xv, yv)
    dz = functionGradient.(Rosenbrock, xv[:], yv[:])

    zv = reshape(zv, n, n)
    contour(v, v, zv, fill=true)
    display(quiver!(xv[:], yv[:], gradient=dz))
end


function testSin()
    range =  0:π/360:2*π
    zv = functionValue.(sin, range)
    dz = functionGradient.(sin, range)

    # display(dz)
    plot(range, zv, label="values")
    display(plot!(range, dz, label = "derivative"))
end


function testCos()
    range =  0:π/360:2*π
    zv = functionValue.(cos, range)
    dz = functionGradient.(cos, range)

    # display(zv)
    plot(range, zv, label="values")
    display(plot!(range, dz, label = "derivative"))
end


function testTan()
    range =  -π/2+0.1:π/360:π/2 - 0.1
    zv = functionValue.(tan, range)
    dz = functionGradient.(tan, range)

    display(zv)
    plot(range, zv, label="values")
    display(plot!(range, dz, label = "derivative"))
end


function testReLu()
    x = -1.0:0.05:+1.0
    # display(ReLu(Variable(-1.0)))
    # display(ReLu.(V)
    zv = functionValue.(ReLu, x)
    dz = functionGradient.(ReLu, x)

    display(zv)
    display(dz)
    plot(x, zv, label="values")
    display(plot!(x, dz, label="derivative"))

end




function testJacobian()
    x = [i for i in -1.0:0.5:1];
    range = [i for i in 0:π/360:2*π] ;
    rangeTan = [i for i in -π/2+π/180:π/180:π/2- π/180];

    v = -1:0.2:+1
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)


    display("Jacobi Rosenbrock")
    dx, dy = jacobian(Rosenbrock, xv, yv)
    display(dx)
    # @show y

    display("Jacobi Relu")
    y = jacobian(ReLu,x);
    display(y)

    display("Jacobi Relu")
    y = jacobian(ReLu,x);
    display(y)

    display("Jacobi Sin")
    y = jacobian(sin, range);
    display(y)
    # display(@benchmark $jacobian(x -> $sin.(x), $range))

    # Jacobi - Cos
    display("Jacobi Cos")
    y = jacobian(cos, range);
    display(y)
    # display(@benchmark $jacobian(x -> $cos.(x), $range))

    # Jacobi - Tan
    display("Jacobi Tan")
    y = jacobian(tan, rangeTan);
    display(y)
end


function main()
    testRosenBrock()
    testSin()
    testCos()
    testTan()
    testReLu()
    testJacobian()
end


main()
