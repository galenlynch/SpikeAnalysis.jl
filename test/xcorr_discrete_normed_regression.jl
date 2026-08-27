using Random: MersenneTwister, randexp
using Statistics: mean, std

"""Generate a homogeneous Poisson spike train via exponential ISIs."""
function poisson_train(rng, rate, dur)
    spikes = Float64[]
    t = -log(rand(rng)) / rate  # first spike time
    while t < dur
        push!(spikes, t)
        t += -log(rand(rng)) / rate
    end
    spikes
end

@testset "xcorr_discrete_normed flat baseline for independent Poisson" begin
    rng = MersenneTwister(123)
    n_trials = 50
    dur = 5.0
    rate = 20.0
    binsize = 0.001
    maxdiff = 0.250
    durs = fill(dur, n_trials)

    us = [poisson_train(rng, rate, dur) for _ in 1:n_trials]
    vs = [poisson_train(rng, rate, dur) for _ in 1:n_trials]

    counts, centers = xcorr_discrete_normed(
        us, vs, durs;
        binsize = binsize,
        maxdiff = maxdiff,
        edgecorrect = true,
        closed = :left,
    )

    @test mean(counts) ≈ 0.0 atol = 0.005

    center_mask = abs.(centers) .< 0.025
    edge_mask = abs.(centers) .> 0.200
    center_mean = mean(counts[center_mask])
    edge_mean = mean(counts[edge_mask])
    @test abs(center_mean - edge_mean) < 0.005
end

@testset "xcorr_discrete_normed edge bin completeness for independent Poisson" begin
    rng = MersenneTwister(123)
    n_trials = 50
    dur = 5.0
    rate = 20.0
    binsize = 0.001
    maxdiff = 0.250
    durs = fill(dur, n_trials)

    us = [poisson_train(rng, rate, dur) for _ in 1:n_trials]
    vs = [poisson_train(rng, rate, dur) for _ in 1:n_trials]

    counts, _ = xcorr_discrete_normed(
        us, vs, durs;
        binsize = binsize,
        maxdiff = maxdiff,
        edgecorrect = true,
        closed = :left,
    )

    n = length(counts)
    outer_std = std(vcat(counts[1:10], counts[(n-9):n]))
    inner_std = std(counts[11:(n-10)])
    ratio = outer_std / inner_std
    @test 0.5 <= ratio <= 2.0
end

@testset "xcorr_discrete_normed symmetry" begin
    rng = MersenneTwister(456)
    n_trials = 10
    dur = 2.0
    rate = 15.0
    durs = fill(dur, n_trials)

    us = [poisson_train(rng, rate, dur) for _ in 1:n_trials]
    vs = [poisson_train(rng, rate, dur) for _ in 1:n_trials]

    counts_uv, centers_uv = xcorr_discrete_normed(us, vs, durs; maxdiff = 0.100)
    counts_vu, centers_vu = xcorr_discrete_normed(vs, us, durs; maxdiff = 0.100)

    @test centers_uv == centers_vu
    # Reversing u and v flips the lag axis
    @test counts_uv ≈ reverse(counts_vu) atol = 1f-6
end

@testset "xcorr_discrete_normed known fixed-delay connection" begin
    # Unit B fires exactly 3ms after each spike of unit A.
    # diff = u - v = A_time - (A_time + 0.003) = -0.003 => peak at lag -3ms.
    rng = MersenneTwister(789)
    n_trials = 20
    dur = 5.0
    rate_a = 10.0
    delay = 0.003
    noise_rate = 5.0
    binsize = 0.001
    maxdiff = 0.050
    durs = fill(dur, n_trials)

    us = Vector{Vector{Float64}}(undef, n_trials)  # unit A
    vs = Vector{Vector{Float64}}(undef, n_trials)  # unit B

    for i in 1:n_trials
        a_spikes = poisson_train(rng, rate_a, dur)
        # B = A shifted by delay, clipped to [0, dur), plus independent noise
        b_shifted = filter(t -> t < dur, a_spikes .+ delay)
        b_noise = poisson_train(rng, noise_rate, dur)
        b_spikes = sort(unique(vcat(b_shifted, b_noise)))
        us[i] = a_spikes
        vs[i] = b_spikes
    end

    counts, centers = xcorr_discrete_normed(
        us, vs, durs;
        binsize = binsize,
        maxdiff = maxdiff,
        edgecorrect = true,
        closed = :left,
    )

    peak_idx = argmax(counts)
    peak_lag = centers[peak_idx]
    # B lags A by `delay`, and `us` is the reference, so the peak sits at
    # +delay.  This asserted -delay while the implementation took u - v.
    @test peak_lag ≈ delay atol = binsize

    # Peak should stand clearly above baseline
    baseline = mean(counts[abs.(centers) .> 0.020])
    @test counts[peak_idx] > baseline + 3 * std(counts[abs.(centers) .> 0.020])
end


@testset "xcorr_discrete_normed identical trains give unit zero lag" begin
    # Self-consistency for the normalized statistic: a unit correlated with
    # its own copy must give exactly 1 at zero lag.  The auto term is the
    # autocorrelogram's own two-sided centre bin, the same quantity the
    # numerator's centre bin is; the earlier one-sided count against a
    # doubled expectation put this between 1.016 and 1.107.
    #
    # Parameterized over train statistics because the candidate corrections
    # disagree in different regimes.
    for kind in (:poisson, :regular, :bursty, :refractory)
        for binsize in (0.0005, 0.002)
            rng = MersenneTwister(5)
            per = Vector{Vector{Float64}}()
            for _ = 1:6
                if kind === :poisson
                    t = sort(rand(rng, 40))
                elseif kind === :regular
                    t = collect(range(0.01, 0.99, length = 40))
                elseif kind === :bursty
                    seeds = sort(rand(rng, 8) .* 0.9)
                    t = sort(vcat([s .+ randexp(rng, 5) .* 0.004 for s in seeds]...))
                    t = t[t .< 1.0]
                else
                    t = sort(rand(rng, 60))
                    t = t[[true; diff(t) .> 0.005]]
                end
                push!(per, t)
            end
            counts, centers = xcorr_discrete_normed(
                per, per, fill(1.0, length(per));
                binsize = binsize, maxdiff = 0.02, edgecorrect = true,
            )
            mid = cld(length(counts), 2)
            @test centers[mid] ≈ 0 atol = 1e-12
            # Float32 accumulation, so eps(Float32) is the achievable floor.
            @test counts[mid] ≈ 1 atol = 1e-6
        end
    end
end
