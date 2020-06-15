module FDStructure

    export Dual, show, value, partials, ReLu, softmax, jacobian, Rosenbrock

    # necessary imports
    import Base: +, -, *, /
    import Base: abs, sin, cos, tan, exp, sqrt, isless
    import Base: convert, promote_rule
    import Base: show

    # structure definition
    struct Dual{T <:Number} <:Number
        v::T
        dv::T
    end

    # operators definition
    -(x::Dual) = Dual(-x.v, -x.dv)
    +(x::Dual, y::Dual) = Dual(x.v + y.v , x.dv + y.dv)
    -(x::Dual, y::Dual) = Dual( x.v - y.v, x.dv - y.dv)
    *(x::Dual, y::Dual) = Dual( x.v * y.v, x.dv * y.v + x.v * y.dv)
    /(x::Dual, y::Dual) = Dual( x.v / y.v, (x.dv * y.v - x.v * y.dv)/y.v^2)

    abs(x::Dual) = Dual(abs(x.v),sign(x.v)*x.dv)
    sin(x::Dual) = Dual(sin(x.v), cos(x.v)*x.dv)
    cos(x::Dual) = Dual(cos(x.v),-sin(x.v)*x.dv)
    tan(x::Dual) = Dual(tan(x.v), one(x.v)*x.dv + tan(x.v)^2*x.dv)
    exp(x::Dual) = Dual(exp(x.v), exp(x.v)*x.dv)
    sqrt(x::Dual) = Dual(sqrt(x.v),0.5/sqrt(x.v) * x.dv)
    isless(x::Dual, y::Dual) = x.v < y.v;

    # promotion/conversion rules
    convert(::Type{Dual{T}}, x::Dual) where T =
     Dual(convert(T, x.v), convert(T, x.dv))
    convert(::Type{Dual{T}}, x::Number) where T =
     Dual(convert(T, x), zero(T))
    convert(::Type{Array{Dual{T}}}, x::Array{Number}) where T =
     Array{Dual}.(convert(T, x), zero(T))
    promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} =
     Dual{promote_type(T,R)}

     # functions
     show(io::IO, x::Dual) = print(io, "(", x.v, ") + [", x.dv, "ϵ]\n");
     value(x::Dual) = x.v;
     partials(x::Dual) = x.dv;

     ReLu(x::Dual{Float64}) = x > zero(x) ? x : zero(x)

     softmax(arg::Array{Dual{Float64}}) = exp.(arg) ./ sum(exp.(arg))

     Rosenbrock(x::Dual{Float64}, y::Number) = (1.0 - x*x) + 100.0*(y - x*x)*(y - x*x)
     Rosenbrock(x::Number, y::Dual{Float64}) = (1.0 - x*x) + 100.0*(y - x*x)*(y - x*x)



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

     # function jacobian(f::typeof(softmax), args::Vector{T}) where {T <:Number}
     #     jacobian_columns = Matrix{T}[]
     #     smax_values = [f(Dual(x, 0.0)) for x in args]
     #     for i=1:length(args)
     #         x = T[]
     #         for j=1:length(args)
     #             if i == j
     #                 push!(x, smax_values[i] * (1.0 - smax_values[i]))
     #             else
     #                 push!(x, -1.0 * smax_values[i] * smax_values[j])
     #             end
     #         end
     #         push!(jacobian_columns, x[:,:])
     #     end
     #     hcat(jacobian_columns...)
     # end

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




end
