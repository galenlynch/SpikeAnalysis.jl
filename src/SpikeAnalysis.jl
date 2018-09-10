module SpikeAnalysis

using StatsBase, GLUtilities

export
    xcorr_discrete_normed,
    acorr_discrete_normed,
    event_intervals

include("util.jl")
include("sp_corrs.jl")
end # module
