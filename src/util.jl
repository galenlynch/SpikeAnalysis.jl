using Random: randperm

function randperm_notsame(n::Integer)
    p = randperm(n)
    for i = 1:n
        new_i = p[i]
        if new_i == i
            while new_i == i
                new_i = rand(1:n)
            end
            p[new_i] = i
            p[i] = new_i
        end
    end
    p
end

function mc_twotail_asymm_p(val, nulldist, nnull::Integer = length(nulldist))
    nless = 0
    nmore = 0
    for nullval in nulldist
        nless = ifelse(nullval <= val, nless + 1, nless)
        nmore = ifelse(nullval >= val, nmore + 1, nmore)
    end
    2 * min(nless, nmore) / nnull
end

function make_sym_bins(bextent, binsize)
    n_sidebins = convert(Int, cld(bextent - binsize / 2, binsize))
    edges = ((-n_sidebins-1/2):1:(n_sidebins+1/2)) * binsize
    centers = ((-n_sidebins):1:n_sidebins) * binsize
    edges, centers, n_sidebins
end

function make_onesided_bins(bextent, binsize)
    nbin = convert(Int, cld(bextent, binsize))
    edges = (0:1:nbin) * binsize
    centers = ((0:1:(nbin-1)) .+ 0.5) .* binsize
    edges, centers
end

function hist_density_reps(
    pts::AbstractVector{<:AbstractVector{<:Number}},
    range_b::Number,
    range_e::Number;
    binsize::Number = 0.01,
    closed::Symbol = :left,
)
    nrep = length(pts)
    max_edge = fld((range_e - range_b), binsize) * binsize + range_b
    edges = range_b:binsize:max_edge
    allpts = cat(pts...; dims = 1)
    h = fit(Histogram, allpts, edges; closed = closed)
    normed_weights = h.weights ./ binsize ./ nrep
    normed_weights, edges, h.weights
end

function missing_mean(
    a::AbstractVector{<:AbstractVector{<:Union{Missing,T}}};
    dims = 1,
) where {T}
    na = length(a)
    na == 0 && throw(ArgumentError("a is empty"))
    nel = length(a[1])
    allsame(length, a) || throw(ArgumentError("vectors in a should be same length"))
    out = Vector{Union{Missing,div_type(T)}}(undef, nel)
    for elno = 1:nel
        elsum = 0
        elcnt = 0
        for i = 1:na
            if !ismissing(a[i][elno])
                elsum += a[i][elno]
                elcnt += 1
            end
            if elcnt > 0
                out[elno] = elsum / elcnt
            else
                out[elno] = missing
            end
        end
    end
    out
end

function hist_quantiles_reps(
    pts::AbstractVector{<:AbstractVector{<:Number}},
    range_b::Number,
    range_e::Number;
    binsize::Number = 0.01,
    closed::Symbol = :left,
    qs = [0.25, 0.5, 0.75],
    outtype::Type = Float32,
)
    nrep = length(pts)
    max_edge = cld((range_e - range_b), binsize) * binsize + range_b
    edges = range_b:binsize:max_edge
    nw = length(edges) - 1
    nq = length(qs)
    hs = map(x -> fit(Histogram, x, edges; closed = closed), pts)
    quants = Vector{Vector{outtype}}(undef, nq)
    for qno = 1:nq
        quants[qno] = Vector{outtype}(undef, nw)
    end
    for binno = 1:nw
        frs = map(x -> x.weights[binno] / binsize, hs)
        q = quantile(frs, qs)
        for qno = 1:nq
            quants[qno][binno] = q[qno]
        end
    end
    quants, edges
end
