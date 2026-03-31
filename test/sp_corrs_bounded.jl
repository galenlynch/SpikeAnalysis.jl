using StatsBase: Histogram, fit

function xcorr_discrete_normed_reference(
    us,
    vs,
    durs;
    binsize = 0.005,
    maxdiff = nothing,
    edgecorrect::Bool = true,
    closed = :left,
)
    isnothing(maxdiff) && error("reference test helper requires maxdiff")
    edges, centers, n_sidebins = SpikeAnalysis.make_sym_bins(maxdiff, binsize)
    counts = zeros(Float32, 2 * n_sidebins + 1)
    auto_u_sum = zero(Float32)
    auto_v_sum = zero(Float32)
    for subno = 1:length(us)
        diffs = vec([u - v for u in us[subno], v in vs[subno]])
        h = fit(Histogram, diffs, edges, closed = closed)
        counts .+= h.weights
        if edgecorrect
            SpikeAnalysis.center_cnts_symm!(
                counts,
                length(us[subno]),
                length(vs[subno]),
                binsize,
                durs[subno],
                n_sidebins,
            )
        else
            SpikeAnalysis.center_cnts!(
                counts,
                length(us[subno]),
                length(vs[subno]),
                binsize,
                durs[subno],
            )
        end
        auto_u_sum += SpikeAnalysis.corrected_auto_counts(
            us[subno],
            binsize,
            durs[subno],
            edgecorrect,
            false,
        )
        auto_v_sum += SpikeAnalysis.corrected_auto_counts(
            vs[subno],
            binsize,
            durs[subno],
            edgecorrect,
            false,
        )
    end
    counts ./= (auto_u_sum * auto_v_sum) ^ (1 / 2)
    counts, centers
end

function acorr_discrete_normed_reference(
    us,
    durs;
    binsize = 0.005,
    maxdiff = nothing,
    edgecorrect::Bool = true,
)
    isnothing(maxdiff) && error("reference test helper requires maxdiff")
    edges, centers = SpikeAnalysis.make_onesided_bins(maxdiff, binsize)
    counts = zeros(Float32, length(centers))
    for subno = 1:length(us)
        diffs = [us[subno][j] - us[subno][i] for i in eachindex(us[subno]) for j in (i+1):length(us[subno])]
        h = fit(Histogram, diffs, edges, closed = :left)
        counts .+= h.weights
        if edgecorrect
            SpikeAnalysis.center_cnts_onesided!(
                counts,
                length(us[subno]),
                binsize,
                durs[subno],
                length(centers),
            )
        else
            SpikeAnalysis.center_cnts!(counts, length(us[subno]), binsize, durs[subno])
        end
    end
    npt = mapreduce(length, +, us, init = 0)
    var_u = abs(npt + counts[1])
    counts ./= var_u
    counts, centers
end

function xcorr_discrete_validonly_reference(
    us,
    vs,
    bounds,
    maxdiff;
    binsize = 0.005,
    normalize = true,
)
    edges, centers, n_sidebins = SpikeAnalysis.make_sym_bins(maxdiff, binsize)
    counts = zeros(Float32, 2 * n_sidebins + 1)
    auto_u_sum = zero(Float32)
    auto_v_sum = zero(Float32)
    ntotal = 0
    durs = SpikeAnalysis.measure.(bounds)
    for subno = 1:length(us)
        ib = searchsortedfirst(vs[subno], bounds[subno][1] + maxdiff)
        ie = searchsortedlast(vs[subno], bounds[subno][2] - maxdiff)
        ie < ib && continue
        diffs = [u - v for u in us[subno] for v in view(vs[subno], ib:ie)]
        ntotal += length(us[subno]) * (ie - ib + 1)
        h = fit(Histogram, diffs, edges, closed = :left)
        counts .+= h.weights
        if normalize
            counts .-= SpikeAnalysis.expected_count(length(us[subno]), length(vs[subno]), binsize, durs[subno])
            auto_u_sum += SpikeAnalysis.corrected_auto_counts(us[subno], binsize, durs[subno], false, false)
            auto_v_sum += SpikeAnalysis.corrected_auto_counts(vs[subno], binsize, durs[subno], false, false)
        end
    end
    if normalize
        counts ./= sqrt(auto_u_sum * auto_v_sum)
    end
    counts, centers, ntotal
end

@testset "xcorr_discrete_normed bounded path" begin
    us = [
        [0.2, 1.0, 1.7, 2.5],
        [0.1, 0.9, 1.8],
    ]
    vs = [
        [0.15, 0.95, 1.6, 2.3],
        [0.3, 1.0, 1.4, 2.0],
    ]
    durs = [3.0, 2.5]
    maxdiff = 0.75

    ref_counts, ref_centers = xcorr_discrete_normed_reference(
        us,
        vs,
        durs;
        binsize = 0.1,
        maxdiff = maxdiff,
    )
    new_counts, new_centers = xcorr_discrete_normed(
        us,
        vs,
        durs;
        binsize = 0.1,
        maxdiff = maxdiff,
    )

    @test new_centers == ref_centers
    @test new_counts ≈ ref_counts atol = 1f-6 rtol = 1f-6
end

@testset "xcorr_discrete_normed closed right falls back" begin
    us = [[0.25, 0.75, 1.25]]
    vs = [[0.1, 0.8, 1.2]]
    durs = [2.0]
    maxdiff = 0.5

    ref_counts, ref_centers = xcorr_discrete_normed_reference(
        us,
        vs,
        durs;
        binsize = 0.25,
        maxdiff = maxdiff,
        closed = :right,
    )
    new_counts, new_centers = xcorr_discrete_normed(
        us,
        vs,
        durs;
        binsize = 0.25,
        maxdiff = maxdiff,
        closed = :right,
    )

    @test new_centers == ref_centers
    @test new_counts ≈ ref_counts atol = 1f-6 rtol = 1f-6
end

@testset "acorr_discrete_normed bounded path" begin
    us = [
        [0.2, 1.0, 1.7, 2.5],
        [0.1, 0.9, 1.8],
    ]
    durs = [3.0, 2.5]
    maxdiff = 0.75

    ref_counts, ref_centers = acorr_discrete_normed_reference(
        us,
        durs;
        binsize = 0.1,
        maxdiff = maxdiff,
    )
    new_counts, new_centers = acorr_discrete_normed(
        us,
        durs;
        binsize = 0.1,
        maxdiff = maxdiff,
    )

    @test new_centers == ref_centers
    @test new_counts ≈ ref_counts atol = 1f-6 rtol = 1f-6
end

@testset "xcorr_discrete_validonly bounded path" begin
    us = [
        [0.2, 1.0, 1.7, 2.5],
        [0.1, 0.9, 1.8],
    ]
    vs = [
        [0.15, 0.95, 1.6, 2.3],
        [0.3, 1.0, 1.4, 2.0],
    ]
    bounds = [(0.0, 3.0), (0.0, 2.5)]
    maxdiff = 0.75

    ref_counts, ref_centers, ref_ntotal = xcorr_discrete_validonly_reference(
        us,
        vs,
        bounds,
        maxdiff;
        binsize = 0.1,
    )
    new_counts, new_centers, new_ntotal = xcorr_discrete_validonly(
        us,
        vs,
        bounds,
        maxdiff;
        binsize = 0.1,
    )

    @test new_centers == ref_centers
    @test new_counts ≈ ref_counts atol = 1f-6 rtol = 1f-6
    @test new_ntotal == ref_ntotal
end
