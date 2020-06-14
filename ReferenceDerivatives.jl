module ReferenceDerivatives

    import Base: cos, sin, tan, sec, one

    ReLu(x) = x > zero(x) ? x : zero(x)
    softmax(arg) = exp.(arg) ./ sum(exp.(arg))

    function dsindx(x)
        cos.(x)
    end

    function dcosdx(x)
        -1.0 .* sin.(x)
    end

    function dtandx(x)
        -1.0 .* sin.(x)
    end

    function dReLudx(x::T) where T
        x > zero(x) ? one(T) : zero(T)
    end

end


# @show n = 5.2
# @show v = [3.1,3.3]
# @show [n...]
# @show [v...]
# @show typeof(v)
# @show v
# @show typeof(v[2,:])
# @show size(v)
#
# x = [i for i in -1.0:0.5:1];
# y = [i for i in -2.0:0.5:0];
# z = [x y]
# @show size(z)[1]
# @show typeof(z)
# @show z
# @show z[1, 2]

# for col in eachcol(x)
#     println(col)
# end
# println.(z)
# x = collect(-10:1:10)
# display(ReLuDerivative.(x))
