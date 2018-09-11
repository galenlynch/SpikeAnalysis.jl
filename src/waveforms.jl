function spike_clip_view(
    signal::AbstractVector{<:Real},
    fs::Real,
    spike_time::Real,
    half_window::Real,
    signal_offset::Real = 0,
    nsig = length(signal)
)
    ndx_b = clip_ndx(
        t_to_ndx(spike_time - half_window, fs, signal_offset),
        nsig
    )
    ndx_e = clip_ndx(
        t_to_ndx(spike_time + half_window, fs, signal_offset),
        nsig
    )
    view(signal, ndx_b:ndx_e)
end

"Get subsection of signal centered at spike_idx, fill empty spots with NaN"
function spike_clip_fixed(
    signal::AbstractVector{<:Number},
    spike_idx::Integer,
    half_basis::Integer,
    nsig::Integer = length(signal)
)
    nbasis = 2 * half_basis + 1
    clip = Vector{Float32}(undef, nbasis)
    idx_b = clip_ndx(spike_idx - half_basis, nsig)
    idx_e = clip_ndx(spike_idx + half_basis, nsig)
    samps_missing_pre = (half_basis - (spike_idx - 1))
    samps_missing_post = (half_basis - (nsig - spike_idx))
    clip[1:samps_missing_pre] .= NaN
    clip_d_idx_b = max(1, samps_missing_pre + 1)
    clip_d_idx_e = min(nbasis, nbasis - samps_missing_post)
    n_clip_d = clip_d_idx_e - clip_d_idx_b + 1
    copyto!(clip, clip_d_idx_b, signal, idx_b, n_clip_d)
    clip[(clip_d_idx_e + 1):end] .= NaN
    clip
end
