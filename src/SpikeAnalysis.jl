module SpikeAnalysis

using StatsBase, GLUtilities, Destruct, Interpolations

export
    xcorr_discrete_normed,
    acorr_discrete_normed,
    event_intervals,
    empirical_qq,
    MarkedInterval,
    raster_syll_data,
    motif_events,
    motif_info,
    align_events,
    align_intervals,
    hist_density_reps,
    spike_clip_fixed

include("util.jl")
include("sp_corrs.jl")
include("song_motifs.jl")
include("waveforms.jl")
end # module
