include("./RDStructure.jl")
using .RDStructure: Node, LeafNode, Variable, ComputableNode, CachedNode,
                   forward, backward, gradient, value, ReLu, fval, fgrad
using Plots


rosenbrock(x, y) = (Variable(1.0) - x*x) + Variable(100.0)*(y - x*x)*(y - x*x)


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


function jacobian(f::Function, args::Vector{T}) where {T <:Number}
    jacobian_columns = Matrix{T}[]
    for i=1:length(args)
        x = T[]
        for j=1:length(args)
            if i == j
                push!(x, fgrad(f, args[j]))
            else
                push!(x, 0.0::T)
            end
        end
        push!(jacobian_columns, x[:,:])
    end
    hcat(jacobian_columns...)
end

function jacobian(f::Function, xargs::Vector{T}, yargs::Vector{T}) where {T <:Number}
    xjacobian_columns = Matrix{T}[]
    yjacobian_columns = Matrix{T}[]
    @assert length(xargs) == length(yargs)
    for i=1:length(xargs)
        x = T[]
        y = T[]
        for j=1:length(xargs)
            if i == j
                xval, yval = fgrad(f, xargs[j], yargs[j])
                push!(x, xval)
                push!(y, yval)
            else
                push!(x, 0.0::T)
                push!(y, 0.0::T)
            end
        end
        push!(xjacobian_columns, x[:,:])
        push!(yjacobian_columns, y[:,:])
    end
    hcat(xjacobian_columns...)
    hcat(yjacobian_columns...)
    xjacobian_columns, yjacobian_columns
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
    dx, dy = jacobian(rosenbrock, xv, yv)
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
    # testRosenBrock()
    # testSin()
    # testCos()
    # testTan()
    # testReLu()
    testJacobian()
end


main()
