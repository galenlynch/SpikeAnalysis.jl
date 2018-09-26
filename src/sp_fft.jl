function blackman_pt(t::Number, dur::Number)
    phase = 2 * pi * t / dur
    0.42 - 0.5 * cos(phase) + 0.08 * cos(2 * phase)
end

function dft_point!(
    ptfft::AbstractVector{Complex{T}},
    pts::AbstractVector{T},
    dur::Real,
    nfft::Integer;
    fbasis::Union{AbstractVector, AbstractRange} = make_fbasis(nfft, T <: Real),
    taperfun::Function = blackman_pt
) where T
    npt = length(pts)
    nf = length(fbasis)
    @inbounds @simd for i = 1:npt
        pt_tap = taperfun(pts[i], dur)
        for j = 1:nf
            ptfft[j] += pt_tap * exp(-im * fbasis[j] * pts[i])
        end
    end
    nothing
end

function dft_point(
    pts::AbstractVector{T},
    dur::Real,
    nfft::Integer;
    fbasis::Union{AbstractVector, AbstractRange} = make_fbasis(nfft, T <: Real),
    kwargs...
) where T
    ptfft = zeros(Complex{Float64}, length(fbasis))
    dft_point!(ptfft, pts, dur, nfft; fbasis = fbasis, kwargs...)
    ptfft
end

function make_fbasis(nfft::Integer, onesided::Bool = true)
    if onesided
        fbasis = 2 * pi * (0:1:div(nfft, 2)) / nfft
    else
        fbasis = 2 * pi * (0:1:(nfft - 1)) / nfft
    end
    fbasis
end

function dft_point_mean_onesided!(
    basis::AbstractVector,
    fft_out::AbstractVector,
    npt::Integer,
    nsamp::Integer,
    winfun::Function,
    pl = plan_rfft(basis)
)
    win = winfun(nsamp)
    norm2 = sum(abs2, win)
    if nsamp > 1
        copyto!(basis, 1, win, 1, nsamp)
    else
        basis[1] = 1
    end
    basis[(nsamp + 1):end] .= 0
    mul!(fft_out, pl, basis)
    fft_out .= npt .* fft_out ./ nsamp
    norm2
end
function dft_point_mean_onesided(
    npt::Integer,
    nsamp::Integer,
    winfun::Function,
    nfft::Integer = nextpow(2, nsamp),
    pl = nothing
)
    basis = Vector{Float64}(undef, nfft)
    fft_out = Vector{Complex{Float64}}(undef, div(nfft, 2) + 1)
    pl = pl == nothing ? plan_rfft(basis) : pl
    norm2 = dft_point_mean_onesided!(basis, fft_out, npt, nsamp, winfun, pl)
    fft_out, norm2
end

function point_psd(
    pts::AbstractVector{T},
    dur::Real,
    fs::Real;
    nfft = nothing
) where T<:Real
    nsamp = ceil(Int, dur * fs)
    if nfft == nothing
        nfft = nextpow(2, nsamp)
    end
    fbasis = fs * 2 * pi * (0:1:div(nfft, 2)) / nfft
    nf = length(fbasis)
    ptfft = dft_point(pts, dur, nfft; fbasis = fbasis)
    winfft, norm2 = dft_point_mean_onesided(length(pts), nsamp, blackman, nfft)
    ptfft .-= winfft
    p = fft2pow!(zeros(T, nf), ptfft, nfft, fs * norm2, true)
    freqs = fbasis / (2 * pi)
    p, freqs
end

function point_psd(
    pt_sets::AbstractVector{<:AbstractVector{T}},
    durs::AbstractVector{<:Real},
    fs;
    nfft = nothing
) where T
    nsamps = ceil.(Int, durs * fs)
    if nfft == nothing
        nfft = nextpow(2, maximum(nsamps))
    end
    fbasis = fs * 2 * pi * (0:1:div(nfft, 2)) / nfft
    nf = length(fbasis)
    n_ptset = length(pt_sets)
    ptfft = zeros(Complex{T}, nf)
    tap_vals = Vector{T}(undef, nfft)
    tap_ffts = Vector{Complex{T}}(undef, nf)
    pl = plan_rfft(tap_vals)
    p = zeros(T, nf)
    for i = 1:n_ptset
        dft_point!(
            ptfft, pt_sets[i], durs[i], nfft; fbasis = fbasis
        )
        norm2 = dft_point_mean_onesided!(
            tap_vals, tap_ffts, length(pt_sets[i]), nsamps[i], blackman, pl
        )
        ptfft .-= tap_ffts
        fft2pow!(p, ptfft, nfft, norm2 * fs, true)
        ptfft .= 0
    end
    p ./= sum(durs)
    freqs = fbasis / (2 * pi)
    p, freqs
end

function point_psd_bin(
    pt_sets::AbstractVector{<:AbstractVector{T}},
    durs::AbstractVector{<:Real},
    fs::Real;
    nfft = nothing
) where T
    nsamps = ceil.(Int, durs * fs)
    if nfft == nothing
        nfft = nextpow(2, maximum(nsamps))
    end
    pds = zeros(T, div(nfft, 2) + 1)
    f = nothing
    for i = 1:length(pt_sets)
        h = fit(Histogram, pt_sets[i], (0:(1/fs):durs[i]), closed = :left)
        pd = mt_pgram(h.weights .- mean(h.weights), fs = fs, nfft = nfft)
        pds .+= durs[i] * pd.power
        f = f == nothing ? pd.freq : f
    end
    pds ./= sum(durs)
    pds, f
end
