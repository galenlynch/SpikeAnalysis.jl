module SpikeAnalysis

import DSP.Periodograms.fft2pow!

using
    StatsBase,
    GLUtilities,
    Destruct,
    Interpolations,
    FFTW,
    DSP,
    LinearAlgebra,
    Statistics,
    PointProcesses,
    Distributed,
    SharedArrays

export
    xcorr_discrete_normed,
    acorr_discrete_normed,
    event_intervals,
    empirical_qq,
    raster_syll_data,
    motif_events,
    motif_info,
    align_events,
    align_intervals,
    hist_density_reps,
    spike_clip_fixed,
    dft_point,
    point_psd,
    point_psd_bin,
    select_motif_sylls,
    align_events,
    align_intervals,
    find_motifs,
    TrigSet,
    RasterTrigSylls,
    trig_data,
    align_trigs,
    group_align_events,
    clip_trigs,
    call_bouts,
    ifr!,
    ifr,
    ifr_upper!,
    ifr_upper,
    missing_mean,
    piecewise_pt_warp!,
    piecewise_pt_warp,
    circ_xcorr_normed,
    xcorr_basis,
    circ_xcorr_unwrap!,
    circ_xcorr_unwrap,
    func_warp_remove!,
    func_warp_remove,
    func_remove_mean!,
    func_remove_mean,
    grow_intervals,
    add_pres,
    circ_xcorr_sig,
    circ_xcorr_null_mc,
    sound_loudness,
    song_rhythm_power_ratio,
    guess_if_array_contains_song,
    score_array_for_song

include("util.jl")
include("sp_corrs.jl")
include("corrs.jl")
include("sp_fft.jl")
include("song_motifs.jl")
include("waveforms.jl")
include("ifr.jl")
include("song_spikes.jl")
end # module
