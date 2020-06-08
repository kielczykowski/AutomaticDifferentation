module FDStructure

    export Dual, show, value, partials, ReLu

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
    promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} =
     Dual{promote_type(T,R)}

     # functions
     show(io::IO, x::Dual) = print(io, "(", x.v, ") + [", x.dv, "ϵ]\n");
     value(x::Dual) = x.v;
     partials(x::Dual) = x.dv;

     ReLu(x::Dual) = x > zero(x) ? x : zero(x)


end
