module DCFI

export load_image_series, register, register_integrate, save_image

import Images: load, save, clamp01nan, imgradients, Gray, KernelFactors
import LinearAlgebra: svd, Diagonal
import Base: round, /, *, -, +
import Statistics: mean
import StaticArrays: SVector
using RegisterDeformation

include("registration.jl")
include("nonrigid.jl")

function load_image_series(images::AbstractVector{String})
    first_image = Float64.(load(images[1]))::Matrix{Float64}
    image_cube = Array{Float64}(undef, size(first_image)..., length(images))
    image_cube[:,:,1] .= first_image

    for (idx,im) in enumerate(images[2:end])
        image_cube[:,:,idx+1] .= Float64.(load(im))::Matrix{Float64}
    end

    image_cube
end

function save_image(filepath::String, images::AbstractArray{<:Real})
    ndims(images) != 3 && throw(ArgumentError("Argument has the wrong number of dimensions (should be 3)"))
    base_fn, ext = splitext(filepath)
    for (i,image) in enumerate(eachslice(images, dims=3))
        save("$base_fn$i$ext", map(clamp01nan, image))
    end
end

function save_image(filepath::String, image::AbstractMatrix{<:Real})
    save("$filepath", map(clamp01nan, image))
end

end #module
