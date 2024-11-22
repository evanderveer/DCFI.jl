module DCFI

using Images
using SubpixelRegistration

function load_image_series(images::AbstractVector{String})
    first_image = Float64.(Gray.(load(images[1])))
    image_cube = Array{Float64}(undef, size(first_image)..., length(images))
    image_cube[:,:,1] .= first_image

    for (idx,im) in enumerate(images[2:end])
        image_cube[:,:,idx+1] .= Float64.(Gray.(load(im)))
    end
    image_cube
end

function register_integrate_absolute!(image_cube::Array{<:Real}; upsample_factor=10)
    coregister!(image_cube, dims=3, upsample_factor=upsample_factor)
    output_image = similar(image_cube[:,:,1])
    num_images = size(image_cube, 3)
    for coord in CartesianIndices(output_image)
        for im in 1:num_images
            output_image[coord] += image_cube[coord, im]
        end
    end
    output_image ./ num_images
end

function register_integrate_absolute(image_cube::Array{<:Real}; kwargs...)
    register_integrate_absolute!(copy(image_cube); kwargs...)
end

function register_integrate_relative!(image_cube::Array{<:Real}; upsample_factor=10)

    num_images = size(image_cube, 3)
    cumulative_phase_offsets = []
    for im in 2:num_images
        curr_phase_offset = phase_offset(image_cube[:,:,im-1], image_cube[:,:,im]; upsample_factor=upsample_factor)
        push!(cumulative_phase_offsets, curr_phase_offset .+ cumulative_phase_offsets[end])
    end

    output_image = similar(image_cube[:,:,1])
    
    for coord in CartesianIndices(output_image)
        for im in 1:num_images
            output_image[coord] += image_cube[coord, im]
        end
    end
    output_image ./ num_images
end

end #module
