# Assumes pts and basis are sorted
function ifr!(out::AbstractVector{<:Union{Missing,Number}}, pts, basis)
    n_out = length(out)
    n_out == length(basis) || throw(ArgumentError("Basis not the same length"))
    last_pt = missing
    last_i = 1
    @inbounds for pt in pts
        i = searchsortedlast(basis, pt)
        val = 1 / (pt - last_pt)
        out[last_i:i] .= val
        if i == n_out
            break
        elseif i > 0
            last_i = i
        end
        last_pt = pt
    end
    if ismissing(last_pt)
        @inbounds out[:] .= missing
    else
        @inbounds out[(last_i+1):end] .= missing
    end
    out
end

function ifr(pts, basis::Union{AbstractVector{T},AbstractRange{T}}) where {T}
    ifr!(Vector{Union{Missing,div_type(T)}}(undef, length(basis)), pts, basis)
end

function ifr_upper!(basis, ifr_in)
    datapred = x -> ! ismissing(x)
    ib = findfirst(datapred, ifr_in)
    ie = findlast(datapred, ifr_in)
    if isnothing(ib) || isnothing(ie)
        ifr_in[:] .= 1 / (basis[end] - basis[1])
    end
    if !isnothing(ib) && ib > 1
        ifr_in[1:(ib-1)] .= 1 / (basis[ib-1] - basis[1])
    end
    if !isnothing(ie) && ie < length(ifr_in)
        ifr_in[(ie+1):end] .= 1 / (basis[end] - basis[ie+1])
    end
    ifr_in
end
ifr_upper(basis, ifr_in) = ifr_upper!(basis, copy(ifr_in))
