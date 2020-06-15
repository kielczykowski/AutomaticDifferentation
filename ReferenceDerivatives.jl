module ReferenceDerivatives

    import Base: cos, sin, tan, sec, one
    export dsindx, dcosdx, dtandx, dReLudx, dSoftmaxdx

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

    function dSoftmaxdx(x::Vector{T}) where {T <: Number}
        derivative_matrix = Matrix{T}[]
        f_value = softmax(x)
        for i =1:length(x)
            col = T[]
            for j =1:length(x)
                if i == j
                    push!(col,f_value[i] * (1.0 - f_value[i]))
                else
                    push!(col, -1.0 * f_value[i] * f_value[j])
                end
            end
            # @show x
            push!(derivative_matrix, col[:,:])
        end
        hcat(derivative_matrix...)
    end

    function dRosenbrockdx(x::Vector{T}, y::Vector{T}) where {T <: Number}

    end

    function dRosenbrockdx(x::Vector{T}, y::Vector{T}) where {T <: Number}

    end


end
