module RDStructure

    export Node, LeafNode, Variable, ComputableNode, CachedNode,
           forward, backward, gradient, value, ReLu, softmax,
           functionValue, functionGradient, Rosenbrock, jacobian


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


    import Base: +, -, *, /, sin, cos, tan, exp, sum, ^, log
    +(x::Node, y::Node) = register(+, x, y)
    -(x::Node, y::Node) = register(-, x, y)
    *(x::Node, y::Node) = register(*, x, y)
    /(x::Node, y::Node) = register(/, x, y)
    ^(x::Node, y::Node) = register(/, x, y)
    sin(x::Node) = register(sin, x)
    cos(x::Node) = register(cos, x)
    tan(x::Node) = register(tan, x)
    exp(x::Node) = register(exp, x)
    # sum(x::Node) = register(sum, x)


    +(x::Variable{Float64}, y::Float64) = +(value(x), y)
    *(x::Variable{Float64}, y::Float64) = *(value(x), y)
    /(x::Variable{Float64}, y::Float64) = /(value(x), y)
    exp(x::Variable{Float64}) = exp(value(x))
    ^(x::Variable{Float64}, y::Float64) = ^(value(x), y)
    # sum(x::Array{Variable{Float64}}) = sum(value.(x))

    +(y::Float64, x::Variable{Float64}) = +(value(x), y)
    *(y::Float64, x::Variable{Float64}) = *(value(x), y)
    /(y::Float64, x::Variable{Float64}) = /(value(x), y)
    ^(y::Float64, x::Variable{Float64}) = ^(value(x), y)

    Rosenbrock(x::Variable{Float64}, y::Variable{Float64}) =
        (Variable(1.0) - x * x) + Variable(100.0) * (y - x * x) * (y - x * x)

    ReLu(x::Variable) = x.value > zero(x.value) ? x : zero(x)

    softmax(arg) = exp.(arg) ./ sum(exp.(arg))

    Base.zero(::Variable{Float64}) = Variable(zero(Float64))


    value(cached::CachedNode) = value(cached.output)
    value(var::Variable) = var.value
    value(var::Float64) = var
    # value(var::Int64) = var



    gradient(::typeof(+), grad, x, y) = (grad, grad)
    gradient(::typeof(-), grad, x, y) = (grad, grad * -1.0)
    gradient(::typeof(/), grad, x, y) = (grad / y, -1.0 * grad * x / y ^ 2)
    gradient(::typeof(*), grad, x, y) = (grad * y, grad * x)
    gradient(::typeof(^), grad, x, y) = (grad * y * x ^ (y-1), grad * x ^ y * log(2.0))
    gradient(::typeof(sin), grad, x) = (grad * cos(x), )
    gradient(::typeof(cos), grad, x) = (grad * -1.0 * sin(x), )
    gradient(::typeof(tan), grad, x) = (grad / (cos(x) ^ 2), )
    gradient(::typeof(exp), grad, x) = (grad * exp(x), )
    # gradient(::typeof(sum), grad, f, x) = (grad * sum(gradient.(f(x))), )

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


    function functionValue(f::Function, xv, yv)
        x, y = Variable(xv), Variable(yv)
        z = f(x, y)
        value(z)
    end


    function functionValue(f::Function, x)
        arg = Variable.(x)
        z = f(arg)
        value.(z)
    end



    function functionGradient(f::Function, xv, yv)
        x, y = Variable(xv), Variable(yv)
        z = f(x, y)
        backward(z, Variable(1.0))
        x.grad, y.grad
    end

    function functionGradient(f::typeof(Rosenbrock), xv, yv)
        x, y = Variable(xv), Variable(yv)
        z = f(x, y)
        backward(z, Variable(1.0))
        5e-4x.grad, 5e-4y.grad
    end


    function functionGradient(f::Function, x)
        arg = Variable(x)
        z = f(arg)
        backward(z, Variable(1.0))
        arg.grad
    end


    function jacobian(f::Function, args::Vector{T}) where {T <:Number}
        jacobian_columns = Matrix{T}[]
        for i=1:length(args)
            x = T[]
            for j=1:length(args)
                if i == j
                    push!(x, functionGradient(f, args[j]))
                else
                    push!(x, 0.0::T)
                end
            end
            push!(jacobian_columns, x[:,:])
        end
        hcat(jacobian_columns...)
    end


    function jacobian(f::typeof(softmax), args::Vector{T}) where {T <:Number}
        jacobian_columns = Matrix{T}[]
        smax_values = functionValue(f, args)
        for i=1:length(args)
            x = T[]
            for j=1:length(args)
                if i == j
                    push!(x, smax_values[i] * (1.0 - smax_values[i]))
                else
                    push!(x, -1.0 * smax_values[i] * smax_values[j])
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
                    xval, yval = functionGradient(f, xargs[j], yargs[j])
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


end
