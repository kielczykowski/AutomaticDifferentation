module RDStructure

    export Node, LeafNode, Variable, ComputableNode, CachedNode,
           forward, backward, gradient, value, ReLu, fval, fgrad


    abstract type Node end
    abstract type LeafNode <: Node end


    mutable struct Variable{T} <: LeafNode
        value::T
        grad::T

        Variable(val::T) where T = new{T}(val, (0.0)::T)
        Variable(val::T, grad::T) where T = new{T}(val, grad)
    end


    struct ComputableNode{TOperation, TAttributes} <: Node
        operation::TOperation
        attribute::TAttributes
    end


    mutable struct CachedNode{TNode, TOutput} <: Node
        node::TNode
        output::TOutput
    end


    function register(op, args...)
        node = ComputableNode(op, args)
        out = forward(node)
        CachedNode(node, out)
    end


    import Base: +, -, *, /, sin, cos, tan
    +(x::Node, y::Node) = register(+, x, y)
    -(x::Node, y::Node) = register(-, x, y)
    *(x::Node, y::Node) = register(*, x, y)
    /(x::Node, y::Node) = register(/, x, y)
    sin(x::Node) = register(sin, x)
    cos(x::Node) = register(cos, x)
    tan(x::Node) = register(tan, x)


    +(x::Variable{Float64}, y::Float64) = +(value(x), y)
    *(x::Variable{Float64}, y::Float64) = *(value(x), y)
    /(x::Variable{Float64}, y::Float64) = /(value(x), y)

    +(y::Float64, x::Variable{Float64}) = +(value(x), y)
    *(y::Float64, x::Variable{Float64}) = *(value(x), y)
    /(y::Float64, x::Variable{Float64}) = /(value(x), y)


    # Base.zero(::Type{Variable{T}}) where T = Variable(zero(T))
    Base.zero(::Variable{Float64}) = Variable(zero(Float64))
    # zero(::Variable{T}) where T = Variable{T}(zero(T))
    # Base.zero(::Type{Variable{T}}) where T = Variable{T}(zero(T))

    ReLu(x::Variable) = x.value > zero(x.value) ? x : zero(x)




    value(cached::CachedNode) = value(cached.output)
    value(var::Variable) = var.value
    value(var::Float64) = var


    gradient(::typeof(+), grad, x, y) = (grad, grad)
    gradient(::typeof(-), grad, x, y) = (grad, grad * -1.0)
    gradient(::typeof(/), grad, x, y) = (grad / y, -1.0 * grad * x / y ^ 2)
    gradient(::typeof(*), grad, x, y) = (grad * y, grad * x)
    gradient(::typeof(sin), grad, x) = (grad * cos(x), )
    gradient(::typeof(cos), grad, x) = (grad * -1.0 * sin(x), )
    gradient(::typeof(tan), grad, x) = (grad / (cos(x) ^ 2), )

    gradient(cached::CachedNode, grad) =
     gradient(cached.node.operation, grad, map(value, cached.node.attribute)...)
    gradient(op::Function, grad, args...) =
     gradient(op, grad, args...)


    forward(var::Variable) = var.value
    forward(op::Function, args...) = op(args...)
    forward(node::LeafNode) = value(node)
    forward(cached::CachedNode) = cached.output = forward(cached.node)
    forward(node::ComputableNode) =
     forward(node.operation,map(forward, node.attribute)...)


    function backward(cached::CachedNode, grad::Any)
      grad_inputs = gradient(cached, grad)
      for (each, each_grad) in zip(cached.node.attribute, grad_inputs)
        backward(each, each_grad)
      end
      nothing
    end


    function backward(var::Variable, grad)
     if isdefined(var, :grad)
            var.grad += grad
        else
            var.grad = grad
        end
        nothing
    end


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
        x.grad, y.grad
    end


    function fgrad(f, x)
        arg = Variable(x)
        z = f(arg)
        backward(z, Variable(1.0))
        # display(arg.grad)
        arg.grad
    end

    # # promotion/conversion rules
    # convert(::Type{Variable{T}}, x::Variable) where T =
    #  Variable(convert(T, x.value), convert(T, x.grad))
    # convert(::Type{Variable{T}}, x::Number) where T =
    #  Variable(convert(T, x), zero(T))
    # convert(::Type{Array{Variable{T}}}, x::Array{Number}) where T =
    #  Array{Variable}.(convert(T, x), zero(T))
    # promote_rule(::Type{Variable{T}}, ::Type{R}) where {T,R} =
    #  Variable{promote_type(T,R)}

end
