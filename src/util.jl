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
