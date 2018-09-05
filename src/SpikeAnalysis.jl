module SpikeAnalysis

using StatsBase, GLUtilities

export xcorr_discrete_normed

include("util.jl")
include("sp_corrs.jl")
end # module
