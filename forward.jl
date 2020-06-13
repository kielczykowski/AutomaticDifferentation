include("./FDStructure.jl")
using .FDStructure: Dual, show, value, partials, ReLu

import Pkg
Pkg.add("Plots")
using Plots

Pkg.add("BenchmarkTools")
using BenchmarkTools


function jacobian(f::Function, args::Vector{T}) where {T <:Number}
    jacobian_columns = Matrix{T}[]
    for i=1:length(args)
        x = Dual{T}[]
        for j=1:length(args)
            seed = (i == j)
            push!(x, seed ?
            Dual(args[j], one(args[j])) :
            Dual(args[j],zero(args[j])) )
        end
        column = partials.([f.(x)...])
        push!(jacobian_columns, column[:,:])
    end
    hcat(jacobian_columns...)
end

function jacobian(f::Function, xargs::Vector{T}, yargs::Vector{T}) where {T <:Number}
    xjacobian_columns = Matrix{T}[]
    yjacobian_columns = Matrix{T}[]
    @assert length(xargs) == length(yargs)
    for i=1:length(xargs)
        x = Dual{T}[]
        y = Dual{T}[]
        for j=1:length(xargs)
            seed = (i == j)
            push!(x, seed ?
            Dual(xargs[j], one(xargs[j])) :
            Dual(xargs[j],zero(xargs[j])) )
            push!(y, seed ?
            Dual(yargs[j], one(yargs[j])) :
            Dual(yargs[j],zero(yargs[j])) )
        end
        xcolumn = partials.([f.(x, yargs)...])
        ycolumn = partials.([f.(xargs, y)...])
        push!(xjacobian_columns, xcolumn[:,:])
        push!(yjacobian_columns, ycolumn[:,:])
    end
    hcat(xjacobian_columns...)
    hcat(yjacobian_columns...)
    xjacobian_columns, yjacobian_columns
end


sigmoid(arg) = 1.0 / (1 + exp(-1.0 * arg))


softmax(arg::Array{Dual{Float64}}) = exp.(arg) ./ sum(exp.(arg))


function testReLu()
    ϵ = Dual(0., 1.)
    x = -1.0:0.05:+1.0
    z = x .+ ϵ
    y = @. ReLu(z)

    plot(x, (p->p.dv).(y), label = "partials")
    display(plot!(x, (p->p.v).(y), label = "value"))

    display(@benchmark $ReLu.($z))
end


function testSin()
    range =  0:π/360:2*π
    sinus = [sin(Dual(x,1.0)) for x in range]
    display(@benchmark $[sin(Dual(x,1.0)) for x in range])

    plot(range, (x->x.v).(sinus), label = "sinus Dual.value")
    display(plot!(range, (x->x.dv).(sinus), label = "sinus Dual.particle"))
end


function testCos()
    range =  0:π/360:2*π
    cosinus = [cos(Dual(x,1.0)) for x in range]
    display(@benchmark $[cos(Dual(x,1.0)) for x in range])

    plot(range, (x->x.v).(cosinus), label = "cosinus Dual.value")
    display(plot!(range, (x->x.dv).(cosinus), label = "cosinus Dual.particle"))
end


function testTan()
    range =  -π/2+π/180:π/180:π/2- π/180
    tang = [tan(Dual(x,1.0)) for x in range]
    display(@benchmark $[tan(Dual(x,1.0)) for x in range])
    plot(range, (x->x.v).(tang), label = "tangens Dual.value")
    display(plot!(range, (x->x.dv).(tang), label = "tangens Dual.particle"))
end


function testRosenbrock()
    rosenbrock(x, y) = (1.0 - x*x) + 100.0*(y - x*x)*(y - x*x)
    v = -1:.2:+1
    n = length(v)
    ϵ = Dual(0., 1.)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)

    z = rosenbrock.(xv .+ ϵ, yv)
    dx = 5e-4partials.(z)
    z = rosenbrock.(xv, yv .+ ϵ)
    dy = 5e-4partials.(z)
    zv = value.(z)

    zv = reshape(zv, n, n)
    contour(v, v, zv, fill=true)
    display(quiver!(xv[:], yv[:], gradient=(dx, dy)))

    display(@benchmark @. $rosenbrock($xv + $ϵ, $yv))
end


function testJacobian()
    x = [i for i in -1.0:0.5:1];
    range = [i for i in 0:π/360:2*π] ;
    rangeTan = [i for i in -π/2+π/180:π/180:π/2- π/180];


    rosenbrock(x, y) = (1.0 - x*x) + 100.0*(y - x*x)*(y - x*x)
    v = -1:.2:+1
    n = length(v)
    xv = repeat(v, inner=n)
    yv = repeat(v, outer=n)

    display("Jacobi Rosenbrock")
    x ,y = jacobian(rosenbrock,xv, yv);
    display(typeofx))
    display(typeof(y))
    # display(@benchmark $jacobian($softmax,$x))

    display("Jacobi Relu")
    x, y = jacobian(ReLu,x);
    display(y)
    # display(@benchmark $jacobian($ReLu,$x))

    # Jacobi - SoftMax

    # display("Jacobi Softmax")
    # y = jacobian(softmax,x);
    # display(@benchmark $jacobian($softmax,$x))

    # Jacobi - Sin
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
    # display(@benchmark $jacobian(x -> $tan.(x), $range))
end

function testSoftmax()

    A = collect(-10.0:0.1:10.0)

    output = []
    ϵ = Dual(0., 1.)
    display(length(A))

    for i in 1:1:length(A)
        list = convert(Array{Dual{Float64}}, deepcopy(A))
        # display(list)
        list[i] = list[i] + ϵ
        # display(i)
        # display(list[i])
        append!(output, softmax(list)[i])
    end

    display("output")
    display(output)
    plot(A, (x->x.v).(output), label = "function values")
    display(plot!(A, (x->x.dv).(output), label = "function derivative"))
end


function main()
    # testReLu()
    # testSin()
    # testCos()
    # testTan()
    # testRosenbrock()
    testJacobian()
    # testSoftmax()

end

main()
