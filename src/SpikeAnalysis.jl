module SpikeAnalysis

using StatsBase, GLUtilities, Destruct, Interpolations

export
    xcorr_discrete_normed,
    acorr_discrete_normed,
    event_intervals,
    empirical_qq

include("util.jl")
include("sp_corrs.jl")
end # module
