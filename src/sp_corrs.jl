# Inputs must be sorted, must be unique
# If points are not unique, normalization will be wrong
"""
    xcorr_discrete_normed(us, vs, durs; binsize=0.005, maxdiff=nothing, edgecorrect=true, closed=:left)

Edge-corrected correlation coefficient cross-correlogram for discrete point
processes observed over multiple subsections (e.g. trials).

# Normalization derivation

The goal is a unitless measure that is zero when the two processes are
independent and conditionally uniform within each subsection.

It is **not** bounded to ``[-1, 1]``.  The estimand is a density, so the
bound is a category error; the value is a shape statistic rather than a
correlation coefficient, and this is why the Python port renamed the mode
from `corrcoef` to `legacy_auto_normalized`.

!!! warning "One duration per subsection, for both trains"
    `durs[s]` is used as the observation window for `us[s]` *and* `vs[s]`,
    and there is no overlap machinery here.  A caller that permutes `vs`
    relative to `us` -- a shift predictor or a trial-shuffle surrogate --
    must therefore ensure the permuted subsections share support with the
    ones they are paired against.  With heterogeneous durations a permuted
    `vs` is silently normalized against the wrong window.  Clip to a common
    window first, or restrict the permutation to subsections of matching
    support.

**Step 1 — raw histogram.**  For each subsection ``s`` with spike trains
``u_s``, ``v_s`` of lengths ``n_u``, ``n_v`` and duration ``d``, accumulate
all pairwise time differences ``u_i - v_j`` into lag bins.  The raw count in
bin ``b`` across all subsections is ``h[b] = \\sum_s h_s[b]``.

!!! warning "Lag sign, and the Python port"
    The difference taken is `u_i - v_j`, so a **positive** lag means a `vs`
    spike came first — `vs` leads `us`. Earlier revisions of this docstring
    said `v_j - u_i`, which is the opposite and does not match the code (see
    `map_pairwise(-, us[subno], vs[subno])` below).

    `aind_ephys_utils.metrics.ccg` uses the opposite convention: its
    `C[i, j]` histograms `t_j - t_i`, so there a positive lag means unit `i`
    leads. The two agree to `eps(Float32)` once one is reversed — verified
    on independent, coupled and identical trains — but a directional
    conclusion does **not** transfer between them unaltered. Reverse the lag
    axis when cross-validating.

**Step 2 — expected count under independence (edge-corrected).**  If spikes
are uniformly distributed within a subsection of duration ``d``, the
probability that a ``(u, v)`` pair falls in a lag bin of width ``w`` whose
far edge is at distance ``binstop`` from zero lag is:

```
P(bin b) = w * (d - binstop + w/2) / d^2
```

The ``(d - binstop + w/2)`` term is the range of reference-spike positions
for which the target spike can land in bin ``b`` without exceeding the
observation window (triangle / edge correction).  For the centre bin
(lag ≈ 0), coincidences arrive from both the positive and negative side,
so the expected count is multiplied by 2 (``center_scale``).

The total expected count is:

```
E[b] = \\sum_s  scale[b] * w[b] * (d_s - binstop[b] + w[b]/2) / d_s^2  *  n_{u,s} * n_{v,s}
```

This respects per-subsection rate variation: subsections where both neurons
fire more contribute proportionally more expected coincidences.

**Step 3 — centred histogram.**  Subtract the expected counts:
``c[b] = h[b] - E[b]``.  Under the null (independent, uniform within each
subsection), ``E[c[b]] = 0``.

**Step 4 — denominator (geometric mean of auto terms).**  To normalize to a
correlation coefficient, divide by ``\\sqrt{A_u \\cdot A_v}`` where each
auto term is:

```
A_u = \\sum_s  [ count\\_auto\\_first(u_s, binsize) - 2 * E_{auto}(s) ]
```

``count_auto_first`` counts spike pairs in ``u_s`` within ``binsize`` of
each other (including self-pairs).  ``E_{auto}`` is the expected auto count
under the same edge-corrected uniform model.  The factor 2 (rather than 1)
accounts for the cross-correlation context: the auto term must match the
variance of the cross-histogram, which sums contributions from both
positive and negative lags of the implicit auto-correlogram at lag 0.

The final normalized correlogram is:

```
C[b] = (h[b] - E[b]) / sqrt(A_u * A_v)
```

# Parameters
- `us::AbstractVector{<:AbstractVector{<:Number}}`: A collection of sets of
  discrete time points (one per subsection / trial).
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
    nbin = 2 * n_sidebins + 1
    counts = zeros(Float32, nbin)
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
                nbin,
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
    halfbin = 1 / (2 * m)  # binsize / 2
    @inbounds for ui in u
        left = ui - maxdiff - halfbin
        right = ui + maxdiff + halfbin
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
    firstbinpairs = 0
    nu = length(u)
    @inbounds for i = 1:istop
        ui = u[i]
        right = ui + maxdiff
        j = i + 1
        while j <= nu && u[j] < right
            d = u[j] - ui
            idx = floor(Int, d * m) + 1
            1 <= idx <= nbin || break
            counts[idx] += 1
            firstbinpairs += idx == 1
            j += 1
        end
    end
    firstbinpairs
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

"""
    corrected_auto_counts(u, binsize, dur, edgecorrect=true, auto=true)

Auto-correlation normalization term for one spike train.

With `auto=true` this is the autocorrelogram's own one-sided count over a
full `binsize`, minus its expected count.

With `auto=false` the term normalizes a *cross*-histogram's centre bin,
which spans `|lag| < binsize/2` and so collects both lag signs. The term
must be that same quantity for the train against itself:
`2 * count_auto_first(u, binsize/2) - nu` is exactly that bin's raw count,
since `count_auto_first` already includes the `nu` self-pairs, so doubling
for the two signs and removing one copy leaves them counted once. The
expectation subtracted is then the centre bin's own — the same
`2 * expected_count_edge_corrected(halfbin, nu, nu, halfbin, dur)` that
`center_cnts_symm!` removes from that bin.

Previously this branch took the one-sided full-width count and doubled the
*expectation* instead, which is a different correction: two units with
identical spike trains gave a zero-lag value of 1.016 to 1.107 rather than 1.
"""
function corrected_auto_counts(u, binsize, dur, edgecorrect::Bool = true, auto::Bool = true)
    nu = length(u)
    if !auto
        halfbin = binsize / 2
        basecount = 2 * count_auto_first(u, halfbin) - nu
        if edgecorrect
            return basecount -
                   2 * expected_count_edge_corrected(halfbin, nu, nu, halfbin, dur)
        end
        # The centre bin is `binsize` wide in total, so no doubling.
        return basecount - expected_count(nu, nu, binsize, dur)
    end
    basecount = count_auto_first(u, binsize)
    if edgecorrect
        corrected_cnt =
            basecount - expected_count_edge_corrected(binsize, nu, binsize, dur)
    else
        corrected_cnt = basecount - expected_count(u, binsize, dur)
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

"""
    expected_count_edge_corrected(binstop, nu, nv, binsize, dur)

Expected coincidence count in a lag bin under independence with edge
correction.  `binstop` is the distance from lag 0 to the far edge of the
bin.  The numerator `binsize * (dur - binstop + binsize/2)` is the area of
the triangle of valid reference positions × bin width; dividing by `dur^2`
gives the probability that a random `(u, v)` pair lands in this bin.
Multiplying by `nu * nv` gives the expected count.
"""
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
        firstbinpairs = forward_hist_windowed!(
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
    nbin = 2 * n_sidebins + 1
    counts = zeros(Float32, nbin)
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
            nbin,
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
