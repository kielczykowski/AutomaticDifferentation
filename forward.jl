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
    column = partials.([f(x)...])
    push!(jacobian_columns, column[:,:])
    end
    hcat(jacobian_columns...)
end

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
    x= [i for i in 1:0.5:10];
    range = [i for i in 0:π/360:2*π] ;
    rangeTan = [i for i in -π/2+π/180:π/180:π/2- π/180];

    display("Jacobi Relu")
    ReLu(x) = x > zero(x) ? x : zero(x)
    y = jacobian(ReLu,x);
    display(@benchmark $jacobian($ReLu,$x))

    # Jacobi - SoftMax

    # display("Jacobi Softmax")
    # y = jacobian(softmax,x);
    # display(@benchmark $jacobian($softmax,$x))

    # Jacobi - Sin
    display("Jacobi Sin")
    y = jacobian(x -> sin.(x), range);
    display(@benchmark $jacobian(x -> $sin.(x), $range))

    # Jacobi - Cos
    display("Jacobi Cos")
    y = jacobian(x -> cos.(x), range);
    display(@benchmark $jacobian(x -> $cos.(x), $range))

    # Jacobi - Tan
    display("Jacobi Tan")
    y = jacobian(x -> tan.(x), rangeTan);
    display(@benchmark $jacobian(x -> $tan.(x), $range))
end


function main()
    # testReLu()
    # testSin()
    # testCos()
    # testTan()
    # testRosenbrock()
    # testJacobian()

end

main()






# # Softmax - do poprawy
#
# softmax(arg) = exp.(arg) ./ sum(exp.(arg))
# # softmax(arg::Dual) = exp(arg)
# softmax(arg::Array{Dual{Float64},1}) = exp.(arg) ./ sum(exp.(arg))
# #! dla każdego innego typu, czyli większych rozmiarów arraya, albo innego typu zmiennych ,  bedzie trzeba napisac funkcję,
# #! żeby wykluczyć pierwszą linijke, ktora akceptuje wszystko. Jak nie ma dosłownego odwolania do danego typu, i nie ma
# #! pierwszej linijki to rzuca błędu bo nie ma definicji funkcji, ktora przyjmuje te argumenty
#
# A = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
# #! pętla po każdym elemencie w wektorze
# A_dual = @.A + ϵ
# softmax(A_dual)
# # println(A_dual);
# # summ(arg) = exp.(arg) ./  sum(exp.(arg))
# # summ(@. A + ϵ)
#
# typeof(A_dual)
# softmax(A_dual)
#
# plot(1:7, (p->p.v).(softmax(A_dual)), label = "value")
# plot!(1:7, (p->p.dv).(softmax(A_dual)), label = "partials")
#
# soft_dual =  softmax(x .+ ϵ)
#
# typeof(soft_dual)
# s= size(soft_dual)
# plot(collect(1:s[1]), (p->p.v).(soft_dual), label = "value")
# plot!(collect(1:s[1]), (p->p.dv).(soft_dual), label = "partial")
#
# @benchmark softmax(x .+ ϵ)
#
