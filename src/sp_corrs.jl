# Inputs must be sorted, must be unique
# If points are not unique, normalization will be wrong
"""
    xcorr_discrete_normed(us, vs, durs; binsize=0.005, maxdiff=nothing, edgecorrect=true, closed=:left)

Calculate the normalized cross-correlation of two sets of discrete time points.

# Parameters
- `us::AbstractVector{<:AbstractVector{<:Number}}`: A collection of sets of
  discrete time points.
- `vs::AbstractVector{<:AbstractVector{<:Number}}`: A collection of sets of
  discrete time points corresponding to `us`.
- `durs::AbstractVector{<:Number}`: Duration of each subsection. Must have the
  same length as `us` and `vs`.
- `binsize::Number=0.005`: Bin size for the histogramming of time differences.
- `maxdiff::Union{Number, Nothing}=nothing`: Maximum time difference to
  consider. If set to `nothing`, it is determined by the maximum of `durs`.
- `edgecorrect::Bool=true`: Whether to apply edge correction to the histogram.
- `closed=:left`: Which side of the bin intervals is closed. Either `:left` or `:right`.

# Returns
- `counts`: Normalized cross-correlation histogram counts.
- `centers`: Centers of the bins in the histogram.

# Notes
- The inputs `us` and `vs` must be sorted and unique. If the points are not
  unique, the normalization will be incorrect.
- Throws an error if `maxdiff` is provided and is less than `binsize`.
- Throws an ArgumentError if the lengths of `us`, `vs`, and `durs` are not the same.
"""
function xcorr_discrete_normed(
    us::AbstractVector{<:AbstractVector{<:Number}},
    vs::AbstractVector{<:AbstractVector{<:Number}},
    durs::AbstractVector{<:Number};
    binsize = 0.005,
    maxdiff = nothing,
    edgecorrect::Bool = true,
    closed = :left,
)
    if !isnothing(maxdiff) && maxdiff < binsize
        error("maxdiff must be bigger than bins")
    end
    n_subsection = length(us)
    if length(vs) != n_subsection || length(durs) != n_subsection
        throw(ArgumentError("us, vs, and durs must have same length"))
    end
    if isnothing(maxdiff)
        bextent = maximum(durs)
    else
        bextent = maxdiff
    end
    edges, centers, n_sidebins = make_sym_bins(bextent, binsize)
    counts = zeros(Float32, 2 * n_sidebins + 1)
    auto_u_sum = zero(Float32)
    auto_v_sum = zero(Float32)
    use_bounded = !isnothing(maxdiff) && closed == :left
    m = 1 / binsize
    firstedge = first(edges)
    for subno = 1:n_subsection
        if use_bounded
            symmetric_hist_windowed!(
                counts,
                us[subno],
                vs[subno],
                maxdiff,
                length(counts),
                m,
                firstedge,
                firstindex(vs[subno]),
                lastindex(vs[subno]),
            )
        else
            diffs = map_pairwise(-, us[subno], vs[subno])
            flatdiffs = reshape(diffs, length(diffs))
            h = fit(Histogram, flatdiffs, edges, closed = closed)
            counts .+= h.weights
        end
        if edgecorrect
            center_cnts_symm!(
                counts,
                length(us[subno]),
                length(vs[subno]),
                binsize,
                durs[subno],
                n_sidebins,
            )
        else
            center_cnts!(counts, length(us[subno]), length(vs[subno]), binsize, durs[subno])
        end
        auto_u_sum +=
            corrected_auto_counts(us[subno], binsize, durs[subno], edgecorrect, false)
        auto_v_sum +=
            corrected_auto_counts(vs[subno], binsize, durs[subno], edgecorrect, false)
    end
    counts ./= sqrt(auto_u_sum * auto_v_sum)
    counts, centers
end

function symmetric_hist_windowed!(
    counts::AbstractVector,
    u::AbstractVector{<:Number},
    v::AbstractVector{<:Number},
    maxdiff::Number,
    nbin::Integer,
    m::Number,
    firstedge::Number,
    lo::Integer,
    vlast::Integer,
)
    hi = lo - 1
    lo > vlast && return 0
    ntotal = 0
    @inbounds for ui in u
        left = ui - maxdiff
        right = ui + maxdiff
        while lo <= vlast && v[lo] < left
            lo += 1
        end
        hi = max(hi, lo - 1)
        while hi < vlast && v[hi+1] <= right
            hi += 1
        end
        lo > hi && continue
        for j = lo:hi
            d = ui - v[j]
            idx = floor(Int, (d - firstedge) * m) + 1
            1 <= idx <= nbin || continue
            counts[idx] += 1
            ntotal += 1
        end
    end
    ntotal
end

function forward_hist_windowed!(
    counts::AbstractVector,
    u::AbstractVector{<:Number},
    maxdiff::Number,
    nbin::Integer,
    m::Number,
    istop::Integer,
)
    ntotal = 0
    firstbinpairs = 0
    @inbounds for i = 1:istop
        j = i + 1
        while j <= length(u) && u[j] < u[i] + maxdiff
            d = u[j] - u[i]
            idx = floor(Int, d * m) + 1
            1 <= idx <= nbin || break
            counts[idx] += 1
            ntotal += 1
            firstbinpairs += idx == 1
            j += 1
        end
    end
    ntotal, firstbinpairs
end

function xcorr_discrete_normed(
    u::AbstractVector{<:Number},
    v::AbstractVector{<:Number},
    dur::Number;
    kwargs...,
)
    xcorr_discrete_normed([u], [v], [dur]; kwargs...)
end

function acorr_discrete_normed(
    us::AbstractVector{<:AbstractVector{<:Number}},
    durs::AbstractVector{<:Number};
    binsize = 0.005,
    maxdiff = nothing,
    edgecorrect::Bool = true,
)
    if !isnothing(maxdiff) && maxdiff < binsize
        error("maxdiff must be bigger than bins")
    end
    n_subsection = length(us)
    if length(durs) != n_subsection
        throw(ArgumentError("us and durs must have same length"))
    end
    if isnothing(maxdiff)
        bextent = maximum(durs)
    else
        bextent = maxdiff
    end
    edges, centers = make_onesided_bins(bextent, binsize)
    n_bin = length(centers)
    counts = zeros(Float32, n_bin)
    use_bounded = !isnothing(maxdiff)
    m = 1 / binsize
    for subno = 1:n_subsection
        if use_bounded
            forward_hist_windowed!(counts, us[subno], maxdiff, n_bin, m, length(us[subno]))
        else
            diffs = map_pairwise(-, us[subno])
            h = fit(Histogram, diffs, edges, closed = :left)
            counts .+= h.weights
        end
        if edgecorrect
            center_cnts_onesided!(counts, length(us[subno]), binsize, durs[subno], n_bin)
        else
            center_cnts!(counts, length(us[subno]), binsize, durs[subno])
        end
    end
    npt = mapreduce(length, +, us, init = 0)
    var_u = abs(npt + counts[1])
    counts ./= var_u
    counts, centers
end

function acorr_discrete_normed(u::AbstractVector{<:Number}, dur::Number; kwargs...)
    acorr_discrete_normed([u], [dur]; kwargs...)
end

function count_auto_first(u, halfbin)
    nu = length(u)
    cnt = nu
    j = 2
    @inbounds for i = 1:nu
        j = max(j, i + 1)
        while j <= nu && u[j] < u[i] + halfbin
            j += 1
        end
        cnt += j - (i + 1)
    end
    cnt
end

function corrected_auto_counts(u, binsize, dur, edgecorrect::Bool = true, auto::Bool = true)
    basecount = count_auto_first(u, binsize)
    count_coeff = ifelse(auto, 1, 2)
    if edgecorrect
        nu = length(u)
        corrected_cnt =
            basecount -
            count_coeff * expected_count_edge_corrected(binsize, nu, binsize, dur)
    else
        corrected_cnt = basecount - count_coeff * expected_count(u, binsize, dur)
    end
    corrected_cnt
end

function center_cnts!(cnts, nu, nv, binsize, dur)
    cnts .-= expected_count(nu, nv, binsize, dur)
end
center_cnts!(cnts, nu, binsize, dur) = center_cnts!(cnts, nu, nu, binsize, dur)

function center_cnts_symm!(cnts, nu, nv, binsize, dur, n_sidebins)
    nc = length(cnts)
    if nc != 2 * n_sidebins + 1
        error("Expected nc to be ", 2 * n_sidebins + 1, " but it is actually ", nc)
    end
    halfbin = binsize / 2
    cnts[cld(nc, 2)] -= 2 * expected_count_edge_corrected(halfbin, nu, nv, halfbin, dur)
    cbin = n_sidebins + 1
    for i = 1:n_sidebins
        e_cnt = expected_count_edge_corrected(halfbin + i * binsize, nu, nv, binsize, dur)
        cnts[cbin+i] -= e_cnt
        cnts[cbin-i] -= e_cnt
    end
    nothing
end

function center_cnts_onesided!(cnts, nu, binsize, dur, n_sidebins)
    for i = 1:n_sidebins
        e_cnt = expected_count_edge_corrected(i * binsize, nu, binsize, dur)
        cnts[i] -= e_cnt
    end
    nothing
end

function expected_count_edge_corrected(binstop, nu, nv, binsize, dur)
    (nu * nv * binsize * (dur - binstop + binsize / 2)) / dur^2
end
function expected_count_edge_corrected(binstop, nu, binsize, dur)
    expected_count_edge_corrected(binstop, nu, nu, binsize, dur)
end

expected_count(nu::Integer, nv::Integer, binsize, dur) = binsize * nu * nv / dur
expected_count(nu::Integer, binsize, dur) = expected_count(nu, nu, binsize, dur)
function expected_count(u::AbstractVector, v::AbstractVector, args...)
    expected_count(length(u), length(v), args...)
end
function expected_count(u::AbstractVector, binsize, dur)
    nu = length(u)
    expected_count(nu, nu, binsize, dur)
end

function event_intervals(a::AbstractVector{<:Number})
    na = length(a)
    na == 0 && return similar(a, 0)
    intervals = similar(a, na - 1)
    @inbounds @simd for i = 1:(na-1)
        intervals[i] = a[i+1] - a[i]
    end
    intervals
end

# Assumes inputs are sorted
function empirical_qq(a::AbstractVector, b::AbstractVector)
    na = length(a)
    nb = length(b)
    if na == nb
        return collect(zip(a, b))
    else
        # Make inverse empirical distribution
        if na > nb
            interp_linear = linear_interpolation((1:na) / na, a)
            outs = Vector{NTuple{2,Float64}}(undef, nb)
            for i = 1:nb
                outs[i] = (interp_linear(i / nb), b[i])
            end
        else
            interp_linear = linear_interpolation((1:nb) / nb, b)
            outs = Vector{NTuple{2,Float64}}(undef, na)
            for i = 1:na
                outs[i] = (a[i], interp_linear(i / na))
            end
        end
    end
    return outs
end

function acorr_discrete_validonly(
    us::AbstractVector{<:AbstractVector},
    bounds::AbstractVector{<:NTuple{2,<:Number}},
    maxdiff::Number;
    binsize = 0.005,
    normalize = true,
)
    n_subsection = length(us)
    durs = measure.(bounds)
    if length(bounds) != n_subsection
        throw(ArgumentError("us, and bounds must have same length"))
    end
    if maxdiff < binsize
        error("maxdiff must be bigger than bins")
    end
    if any(2 * maxdiff .> durs)
        error("2 * maxdiff must be smaller than all durs")
    end
    edges = 0:binsize:maxdiff
    centers = edges[1:(end-1)] .+ binsize / 2
    nbin = length(edges) - 1
    counts = zeros(Float32, nbin)
    auto_u_sum = zero(Float32)
    ntotal = 0
    m = 1 / binsize

    for subno = 1:n_subsection
        ie = searchsortedlast(us[subno], bounds[subno][2] - maxdiff)
        ie == 0 && continue
        l = length(us[subno])
        _, firstbinpairs = forward_hist_windowed!(
            counts,
            us[subno],
            maxdiff,
            nbin,
            m,
            ie,
        )
        ntotal += ie * l - ie * (ie + 1) ÷ 2
        auto_u_sum += ie + firstbinpairs

        if normalize
            ec = expected_count(l, ie, binsize, durs[subno])
            counts .-= ec
            auto_u_sum -= ec
        end
    end

    if normalize
        counts ./= auto_u_sum
    end

    counts, centers, ntotal
end

function acorr_discrete_validonly(us, durs::AbstractVector{<:Number}, maxdiff; kwargs...)
    acorr_discrete_validonly(us, tuple.(zero(eltype(durs)), durs), maxdiff; kwargs...)
end

function acorr_discrete_validonly(
    u::AbstractVector{<:Real},
    dur::Number,
    maxdiff;
    kwargs...,
)
    acorr_discrete_validonly([u], [dur], maxdiff; kwargs...)
end

function xcorr_discrete_validonly(
    us::AbstractVector{<:AbstractVector},
    vs::AbstractVector{<:AbstractVector},
    bounds::AbstractVector{<:NTuple{2,<:Number}},
    maxdiff::Number;
    binsize = 0.005,
    normalize = true,
)
    n_subsection = length(us)
    if length(vs) != n_subsection || length(bounds) != n_subsection
        throw(ArgumentError("us, vs, and bounds must have same length"))
    end
    if maxdiff < binsize
        error("maxdiff must be bigger than bins")
    end
    durs = measure.(bounds)
    if any(2 * maxdiff .> durs)
        error("2 * maxdiff must be smaller than all durs")
    end
    edges, centers, n_sidebins = make_sym_bins(maxdiff, binsize)
    counts = zeros(Float32, 2 * n_sidebins + 1)
    auto_u_sum = zero(Float32)
    auto_v_sum = zero(Float32)
    ntotal = 0
    m = 1 / binsize
    firstedge = first(edges)

    for subno = 1:n_subsection
        nu = length(us[subno])
        nv = length(vs[subno])
        ib = searchsortedfirst(vs[subno], bounds[subno][1] + maxdiff)
        ie = searchsortedlast(vs[subno], bounds[subno][2] - maxdiff)
        symmetric_hist_windowed!(
            counts,
            us[subno],
            vs[subno],
            maxdiff,
            length(counts),
            m,
            firstedge,
            ib,
            ie,
        )
        ntotal += nu * max(0, ie - ib + 1)

        if normalize
            counts .-= expected_count(nu, nv, binsize, durs[subno])
            auto_u_sum +=
                corrected_auto_counts(us[subno], binsize, durs[subno], false, false)
            auto_v_sum +=
                corrected_auto_counts(vs[subno], binsize, durs[subno], false, false)
        end
    end

    if normalize
        auto_prod = auto_u_sum * auto_v_sum
        auto_prod == 0 && error("No points made it!")
        counts ./= auto_prod ^ (1 / 2)
    end

    counts, centers, ntotal
end

function xcorr_discrete_validonly(
    us,
    vs,
    durs::AbstractVector{<:Number},
    maxdiff;
    kwargs...,
)
    xcorr_discrete_validonly(us, vs, tuple.(zero(eltype(durs)), durs), maxdiff; kwargs...)
end

function xcorr_discrete_validonly(
    us::AbstractVector{<:Number},
    vs::AbstractVector{<:Number},
    dur::Number,
    maxdiff::Number;
    kwargs...,
)
    xcorr_discrete_validonly([us], [vs], [dur], maxdiff; kwargs...)
end

function xcorr_discrete_validonly(
    us::AbstractVector{<:Points{<:Any,<:Any,<:Any,<:NakedPoint}},
    vs::AbstractVector{<:Points{<:Any,<:Any,<:Any,<:NakedPoint}},
    maxdiff::Number;
    kwargs...,
)
    bnds = bounds.(us)
    all(bnds .== bounds.(vs)) || error("Bounds do not match")
    xcorr_discrete_validonly(nakedvalues.(us), nakedvalues.(vs), bnds, maxdiff; kwargs...)
end
