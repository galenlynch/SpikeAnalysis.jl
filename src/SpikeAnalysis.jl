module SpikeAnalysis

using StatsBase, GLUtilities

export xcorr_discrete_normed

# Inputs must be sorted, should be unique
function xcorr_discrete_normed(
    u, v, dur;
    binsize = 0.005,
    maxdiff = nothing,
    edgecorrect::Bool = true
)
    diffs = map_pairwise(-, u, v)
    stdu = std_autocorr(u, binsize, dur, edgecorrect)
    stdv = std_autocorr(v, binsize, dur, edgecorrect)
    if maxdiff != nothing
        maxdiff > binsize || error("maxdiff must be bigger than bins")
        ds = filter(x -> abs(x) <= maxdiff, view(diffs, :))
        bextent = maxdiff - binsize
    else
        ds = view(diffs, :)
        bextent = dur
    end
    xcnts, centers, _, _, n_sidebins = hist_points_symm(
        ds, binsize = binsize, bextent = bextent
    )
    xc = convert(Vector{Float64}, xcnts)
    extraargs = edgecorrect ? (n_sidebins) : ()
    center_cnts!(xc, length(u), length(v), binsize, dur, extraargs...)
    xc .= xc ./ (stdu * stdv)
    xc, centers
end

function count_auto_first(u, halfbin)
    length(u) + count(x -> x < halfbin, map_pairwise(-, u))
end

function std_autocorr(u, binsize, dur, edgecorrect::Bool = true)
    halfbin = binsize / 2
    basecount = count_auto_first(u, halfbin)
    if edgecorrect
        corrected_cnt = basecount - expected_count(u, binsize, dur)
    else
        nu = length(u)
        corrected_cnt = basecount - 2 * expected_count_edge_corrected(
            halfbin, nu, nu, halfbin, dur
        )
    end
    abs(corrected_cnt) ^ (1/2)
end

function center_cnts!(cnts, nu, nv, binsize, dur, n_sidebins)
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

function center_cnts!(cnts, nu, nv, binsize, dur)
    cnts .= cnts .- expected_count(nu, nv, binsize, dur)
end

function expected_count_edge_corrected(binstop, nu, nv, binsize, dur)
    (nu * nv * binsize * (dur - binstop + binsize / 2)) / dur^2
end

expected_count(nu::Integer, nv::Integer, binsize, dur) = binsize * nu * nv / dur
function expected_count(u::AbstractVector, v::AbstractVector, args...)
    expected_count(length(u), length(v), args...)
end
function expected_count(u::AbstractVector, binsize, dur)
    nu = length(u)
    expected_count(nu, nu, binsize, dur)
end

function hist_points_symm(
    pts::AbstractVector;
    binsize::Real = 0.005,
    bextent::Union{Real, Nothing} = nothing,
    closed::Symbol = :left
)
    if bextent == nothing
        bextent = maximum(abs.(pts))
    end
    n_sidebins = convert(Int, cld(bextent + binsize / 2, binsize))
    expanded_extent = binsize * (n_sidebins + 0.5)
    edges = -expanded_extent:binsize:expanded_extent
    centers = edges[1:end-1] .+ binsize ./ 2
    h = fit(Histogram, pts, edges, closed = closed)
    h.weights, centers, edges, bextent, n_sidebins
end

end # module
