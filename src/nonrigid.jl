normsquared(grad::SVector{2, <:Real}) = sum(grad .^ 2)
normalize_gradients!(grad_image::Matrix{<:SVector{2, <:Real}}) = grad_image ./= maximum(normsquared.(grad_image))

function gradient_image(image::Matrix{<:Union{<:Real, <:Gray}}, kernelfun=KernelFactors.sobel, boundary="reflect")
    grad = imgradients(image, kernelfun, boundary) #Always returns Matrix{Float64}
    grad_image = Matrix{SVector{2, eltype(grad[1])}}(undef, size(grad[1])...)
    fill_grad_image!(grad_image, grad)
    normalize_gradients!(grad_image)
    grad_image 
end

function gradient_image(cube::Array{T, 3}, kernelfun=KernelFactors.sobel, boundary="reflect") where T<:Real
    grad_cube = Array{SVector{2, T}, 3}(undef, size(cube)...)
    fill_grad_cube!(grad_cube, cube, kernelfun, boundary)
    grad_cube
end

function fill_grad_cube!(grad_cube, cube, kernelfun, boundary)
    for i in axes(grad_cube, 3)
        grad_cube[:,:,i] = gradient_image(cube[:,:,i], kernelfun, boundary)
    end
end

function fill_grad_image!(grad_image, grad::Tuple{AbstractMatrix{<:T}, AbstractMatrix{<:T}}) where T
    for coord in CartesianIndices(grad_image)
        grad_image[coord] = SVector{2, T}(grad[1][coord], grad[2][coord])
    end
end

function image_difference(cube::AbstractArray{<:Real, 3}, static::AbstractMatrix{<:Real}) 
    T = promote_type(eltype(cube), eltype(static))
    diff_cube = Array{T, 3}(undef, size(cube)...)
    for idx in axes(cube, 3)
        diff_cube[:,:,idx] = cube[:,:,idx] .- static
    end
    diff_cube
end

function gradient_image_sum(cube::AbstractArray{<:SVector{2, <:Real}, 3}, static::AbstractMatrix{<:SVector{2, <:Real}}) 
    T = promote_type(eltype(cube), eltype(static))
    sum_cube = Array{T, 3}(undef, size(cube)...)
    for idx in axes(cube, 3)
        sum_cube[:,:,idx] = cube[:,:,idx] .+ static
    end
    sum_cube
end

function incremental_transformation_field(
    cube::AbstractArray{<:Real, 3}, 
    grad_cube::AbstractArray{<:SVector{2, <:Real}, 3}, 
    static::AbstractMatrix{<:Real},
    grad_static::AbstractMatrix{<:SVector{2, <:Real}}
    )

    intensity_diff_cube = -image_difference(cube, static)
    grad_sum_cube = gradient_image_sum(grad_cube, grad_static)
    
    grad_sum_cube .* intensity_diff_cube
end

function row_lock!(trans_field::AbstractMatrix{<:SVector})
    mean_vec = mean(trans_field, dims=2)
    for col in eachcol(trans_field)
        col .= mean_vec
    end
    trans_field
end 

function row_lock!(trans_field::AbstractArray{<:SVector, 3})
    for idx in axes(trans_field, 3)
        trans_field[:,:,idx] .= row_lock!(trans_field[:,:,idx])
    end
    trans_field
end

function field2displacement(trans_field::AbstractMatrix{<:SVector{2, T}}) where T<:Real
    disps = Array{T, 3}(undef, 2, size(trans_field)...)
    
    disps[1,:,:] .= getindex.(trans_field, 1)
    disps[2,:,:] .= getindex.(trans_field, 2)
    disps
end

function image2nodes(image::AbstractMatrix{<:Real})
    nodes = map(axes(image), size(image)) do ax, g
        range(first(ax), stop=last(ax), length=g)
    end
    nodes
end

function register_nonrigid_iter(cube::AbstractArray{<:Real, 3})
    cube_registered = register(cube, max_shift=10)
    static = mean(cube_registered, dims=3)[:,:, firstindex(cube_registered, 3)]
    fom_sum1 = 0
    fom_sum2 = 0

    cube_grad = gradient_image(cube_registered)
    static_grad = gradient_image(static)
    trans_field = incremental_transformation_field(cube_registered, cube_grad, static, static_grad)
    row_lock!(trans_field)
    cube_nonrigid = copy(cube_registered)

    for idx in axes(cube_registered, 3)
        fom_sum1 += sum((cube_registered[:,:,idx] .- static).^2)
        displacements = field2displacement(trans_field[:,:,idx])
        nodes = image2nodes(cube_registered[:,:,idx])
        gd = GridDeformation(displacements, nodes)
        cube_nonrigid[:,:,idx] .= warp(cube_registered[:,:,idx], gd)
        fom_sum2 += sum((map(x->isnan(x) ? 0.0 : x, cube_nonrigid[:,:,idx]) .- static).^2)
    end
    println("FoM before: $fom_sum1 \t After: $fom_sum2")
    map(x->isnan(x) ? 0.0 : x, cube_nonrigid)
end

