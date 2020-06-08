module RDStructure

export Node, LeafNode, Variable, ComputableNode, CachedNode,
       forward, backward, gradient, value

abstract type Node end
abstract type LeafNode <: Node end
mutable struct Variable{T} <: LeafNode
    value::T
    grad::T
#! inicjalizacja wartości gradientu nie wartościami losowymi!!!!
    Variable(val::T) where T = new{T}(val, 0.001 .* val)
    Variable(val::T, grad::T) where T = new{T}(val, 0.001 .*val)
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

import Base: +, -, *, /, sin
+(x::Node, y::Node) = register(+, x, y)
-(x::Node, y::Node) = register(-, x, y)
*(x::Node, y::Node) = register(*, x, y)
/(x::Node, y::Node) = register(/, x, y)
sin(x::Node) = register(sin,x)

forward(cached::CachedNode) = cached.output = forward(cached.node)
forward(node::ComputableNode) = forward(node.operation, map(forward, node.attribute)...)
forward(op::Function, args...) = op(args...)
forward(var::Variable) = var.value

function backward(cached::CachedNode, grad::Any)
  grad_inputs = gradient(cached, grad)
  for (each, each_grad) in zip(cached.node.attribute, grad_inputs)
    backward(each, each_grad)
  end
end
# function backward(cached::CachedNode)
# end
+(x::Variable{Float64}, y::Float64) = +(value(x), y)
*(x::Variable{Float64}, y::Float64) = *(value(x), y)
# -(x::CachedNode{ComputableNode{typeof(+),Tuple{Variable{Float64},Variable{Float64}}},Float64}, y::Float64) =
#     -(value(x),y)
# *(x::CachedNode{ComputableNode{typeof(+),Tuple{Variable{Float64},Variable{Float64}}},Float64}, y::Float64) =
#     *(value(x), y)

gradient(cached::CachedNode, grad) =
 gradient(cached.node.operation, grad, map(value, cached.node.attribute)...)
gradient(op::Function, grad, args...) =
 gradient(op, grad, args...)
value(cached::CachedNode) = value(cached.output)
value(var::Variable) = var.value
value(var::Float64) = var
gradient(::typeof(+), grad, x, y) = (grad * (1 + y), grad * (x+1))
gradient(::typeof(-), grad, x, y) = (grad * (1- y), grad * (x-1))
gradient(::typeof(/), grad, x, y) = (grad / y, -1.0 * grad * x / y^2)
gradient(::typeof(*), grad, x, y) = (grad * y, grad * x)
gradient(::typeof(sin), grad, x) = (grad * cos(x), )



function backward(var::Variable, grad)
 if isdefined(var, :grad)
        var.grad += grad
    else
        var.grad = grad
    end
    nothing
end

end
