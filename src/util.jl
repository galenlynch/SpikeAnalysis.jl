function make_sym_bins(bextent, binsize)
    n_sidebins = convert(Int, cld(bextent - binsize / 2, binsize))
    edges = ((-n_sidebins - 1/2):1:(n_sidebins + 1/2)) * binsize
    centers = (-n_sidebins:1:n_sidebins) * binsize
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
    closed::Symbol =  :left,
)
    nrep = length(pts)
    max_edge = cld((range_e - range_b),  binsize) * binsize + range_b
    edges = range_b:binsize:max_edge
    allpts = cat(pts...; dims = 1)
    h = fit(Histogram, allpts, edges; closed = closed)
    normed_weights = h.weights ./ binsize ./ nrep
    normed_weights, edges
end
