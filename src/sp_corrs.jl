# Inputs must be sorted, must be unique
# If points are not unique, normalization will be wrong
function xcorr_discrete_normed(
    us::AbstractVector{<:AbstractVector{<:Number}},
    vs::AbstractVector{<:AbstractVector{<:Number}},
    durs::AbstractVector{<:Number};
    binsize = 0.005,
    maxdiff = nothing,
    edgecorrect::Bool = true,
    closed = :left
)
    if maxdiff != nothing && maxdiff < binsize
        error("maxdiff must be bigger than bins")
    end
    n_subsection = length(us)
    if length(vs) != n_subsection || length(durs) != n_subsection
        throw(ArgumentError("us, vs, and durs must have same length"))
    end
    if maxdiff == nothing
        bextent = maximum(durs)
    else
        bextent = maxdiff
    end
    edges, centers, n_sidebins = make_sym_bins(bextent, binsize)
    counts = zeros(Float32, 2 * n_sidebins + 1)
    auto_u_sum = zero(Float32)
    auto_v_sum = zero(Float32)
    for subno = 1:n_subsection
        diffs = map_pairwise(-, us[subno], vs[subno])
        flatdiffs = reshape(diffs, length(diffs))
        h = fit(Histogram, flatdiffs, edges, closed = closed)
        counts .+= h.weights
        if edgecorrect
            center_cnts_symm!(
                counts,
                length(us[subno]),
                length(vs[subno]),
                binsize,
                durs[subno],
                n_sidebins
            )
        else
            center_cnts!(
                counts, length(us[subno]), length(vs[subno]), binsize, durs[subno]
            )
        end
        auto_u_sum += corrected_auto_counts(
            us[subno], binsize, durs[subno], edgecorrect, false
        )
        auto_v_sum += corrected_auto_counts(
            vs[subno], binsize, durs[subno], edgecorrect, false
        )
    end
    counts ./= (auto_y_sum * auto_v_sum) ^ (1 / 2)
    counts, centers
end

function  xcorr_discrete_normed(
    u::AbstractVector{<:Number}, v::AbstractVector{<:Number}, dur::Number;
    kwargs...
)
    xcorr_discrete_normed([u], [v], [dur]; kwargs...)
end

function acorr_discrete_normed(
    us::AbstractVector{<:AbstractVector{<:Number}},
    durs::AbstractVector{<:Number};
    binsize = 0.005,
    maxdiff = nothing,
    edgecorrect::Bool = true
)
    if maxdiff != nothing && maxdiff < binsize
        error("maxdiff must be bigger than bins")
    end
    n_subsection = length(us)
    nus = length.(us)
    if length(durs) != n_subsection
        throw(ArgumentError("us and durs must have same length"))
    end
    if maxdiff == nothing
        bextent = maximum(durs)
    else
        bextent = maxdiff
    end
    edges, centers = make_onesided_bins(bextent, binsize)
    fitlermax = edges[end]
    n_bin = length(centers)
    counts = zeros(Float32, n_bin)
    for subno = 1:n_subsection
        diffs = map_pairwise(-, us[subno])
        h = fit(Histogram, diffs, edges, closed = :left)
        counts .+= h.weights
        if edgecorrect
            center_cnts_onesided!(
                counts, length(us[subno]), binsize, durs[subno], n_bin
            )
        else
            center_cnts!(counts, length(us[subno]), binsize, durs[subno])
        end
    end
    npt = mapreduce(length, +, us, init = 0)
    var_u = abs(npt + counts[1])
    counts ./= var_u
    counts, centers
end

function acorr_discrete_normed(
    u::AbstractVector{<:Number}, dur::Number; kwargs...
)
    acorr_discrete_normed([u], [dur]; kwargs...)
end

function count_auto_first(u, halfbin)
    length(u) + count(x -> x < halfbin, map_pairwise(-, u))
end

function corrected_auto_counts(
    u, binsize, dur, edgecorrect::Bool = true, auto::Bool = true
)
    basecount = count_auto_first(u, binsize)
    count_coeff = ifelse(auto, 1, 2)
    if edgecorrect
        nu = length(u)
        corrected_cnt = basecount - count_coeff * t_count_edge_corrected(
            binsize, nu, binsize, dur
        )
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
    cnts[cld(nc, 2)] -= 2 * expected_count_edge_corrected(
        halfbin, nu, nv, halfbin, dur
    )
    cbin = n_sidebins + 1
    for i = 1:n_sidebins
        e_cnt = expected_count_edge_corrected(
            halfbin + i * binsize, nu, nv, binsize, dur
        )
        cnts[cbin + i] -= e_cnt
        cnts[cbin - i] -= e_cnt
    end
    nothing
end

function center_cnts_onesided!(cnts, nu, binsize, dur, n_sidebins)
    for i = 1:n_sidebins
        e_cnt = expected_count_edge_corrected(
            i * binsize, nu, binsize, dur
        )
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
    @inbounds @simd for i = 1:(na - 1)
        intervals[i] = a[i + 1] - a[i]
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
            interp_linear = LinearInterpolation((1:na) / na, a)
            return [(interp_linear(i / nb), x) for (i, x) in enumerate(b)]
        else
            interp_linear = LinearInterpolation((1:nb) / nb, b)
            return [(x, interp_linear(i / na)) for (i, x) in enumerate(a)]
        end
    end
end

function acorr_discrete_validonly(
    us::AbstractVector{<:AbstractVector},
    durs::AbstractVector{<:Number},
    maxdiff::Number;
    binsize = 0.005,
    normalize = true
)
    n_subsection = length(us)
    if length(durs) != n_subsection
        throw(ArgumentError("us, and durs must have same length"))
    end
    if maxdiff < binsize
        error("maxdiff must be bigger than bins")
    end
    if any(2 * maxdiff .> durs)
        error("2 * maxdiff must be smaller than all durs")
    end
    edges = 0:binsize:maxdiff
    centers = edges[1:end-1] .+ binsize / 2
    nbin = length(edges) - 1
    counts = zeros(Float32, nbin)
    auto_u_sum = zero(Float32)
    ntotal = 0
    m = 1 / binsize

    for subno = 1:n_subsection
        ie = searchsortedlast(us[subno], durs[subno] - maxdiff)
        ie == 0 && continue
        auto_u_sum += ie
        l = length(us[subno])
        for i = 1:ie
            for j = i + 1:l
                ntotal += 1
                d = us[subno][j] - us[subno][i]
                d >= maxdiff && break
                _glhist_push!(counts, d, 0, nbin, m)
                auto_u_sum += ifelse(d < binsize, 1, 0)
            end
        end

        if normalize
            ec = expected_count(l, ie, binsize, durs[subno])
            counts .-= ec
            auto_u_sum -= ec
        end
    end

    if normalize
        auto_u_sum == 0 && error("No points made it!")
        counts ./=  auto_u_sum
    end

    counts, centers, ntotal
end

function acorr_discrete_validonly(
    u::AbstractVector{<:Real}, dur::Real, maxdiff; kwargs...
)
    acorr_discrete_validonly([u], [dur], maxdiff; kwargs...)
end


function xcorr_discrete_validonly(
    us::AbstractVector{<:AbstractVector},
    vs::AbstractVector{<:AbstractVector},
    durs::AbstractVector{<:Number},
    maxdiff::Number;
    binsize = 0.005,
    normalize = true
)
    n_subsection = length(us)
    if length(vs) != n_subsection || length(durs) != n_subsection
        throw(ArgumentError("us, vs, and durs must have same length"))
    end
    if maxdiff < binsize
        error("maxdiff must be bigger than bins")
    end
    if any(2 * maxdiff .> durs)
        error("2 * maxdiff must be smaller than all durs")
    end
    edges, centers, n_sidebins = make_sym_bins(maxdiff, binsize)
    counts = zeros(Float32, 2 * n_sidebins + 1)
    auto_u_sum = zero(Float32)
    auto_v_sum = zero(Float32)
    ntotal = 0

    for subno = 1:n_subsection
        ib = searchsortedfirst(vs[subno], maxdiff)
        ie = searchsortedlast(vs[subno], durs[subno] - maxdiff)
        ie < ib && continue
        diffs = imap_product(-, us[subno], view(vs[subno], ib:ie))
        ntotal += length(diffs)

        glhist!(counts, diffs, edges)

        if normalize
            counts .-= expected_count(nu, nv, binsize, durs[subno])
            auto_u_sum += corrected_auto_counts(
                us[subno], binsize, durs[subno], false, false
            )
            auto_v_sum += corrected_auto_counts(
                vs[subno], binsize, durs[subno], false, false
            )
        end
    end

    if normalize
        auto_prod = auto_u_sum * auto_v_sum
        auto_prod == 0 && error("No points made it!")
        counts ./=  auto_prod ^ (1 / 2)
    end

    counts, centers, ntotal
end

function xcorr_discrete_validonly(
    us::AbstractVector{<:Number},
    vs::AbstractVector{<:Number},
    dur::Number,
    maxdiff::Number;
    kwargs...
)
    xcorr_discrete_validonly([us], [vs], [dur], maxdiff; kwargs...)
end
