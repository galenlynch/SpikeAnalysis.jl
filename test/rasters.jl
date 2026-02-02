using Compat,
    GlDbPlotting,
    GLFileCache,
    LibPQ,
    OEUtilities,
    SignalIndices,
    SortedIntervals,
    TSConditioning,
    DSP,
    OpenEphysLoader,
    NeuroPlots,
    OEPlotting,
    DynamicTimeseries,
    PyQtGraph,
    PyCall,
    Phy,
    OEDiscovery,
    DataFrames,
    Distances,
    Missings,
    SpikeAnalysis,
    PyPlot,
    Seaborn,
    Statistics,
    Destruct,
    HypothesisTests,
    StatsBase,
    Optim

const conn_str =  "postgresql://galen@localhost:5433/galen"
const conn = LibPQ.Connection(con(conn, song_rid)
mic_rec_dur = durs_syll_rec[1]

# Get spikes
(
    recording_ids,
    chnos,
    electrode_depths,
    spk_type,
    label_ids,
    labels_spk,
    durs_recs,
    spks,
    x_ch,
    y_ch,
    z_ch
) = get_spikes(conn, song_rid)

# Get whitened files
neur_rids = unique(recording_ids)
neur_dbfiles = map(rid -> dbfiles((file_by_rid(conn, rid, NamedTuple)))[1], neur_rids)
neur_whitened_mmaps = access_filtered_arrays(neur_dbfiles, whitened_files, whitened_lookup)
neur_whitened_map = Dict(zip(neur_rids, neur_whitened_mmaps))

# Get original recordings
original_fnames = find_db_file.(
        neur_dbfiles, lookaside = lookaside, context = context
)
original_neur_recs = SampleArray.(Float32, original_fnames, false)
neur_map = Dict(zip(neur_rids, original_neur_recs))

# Make syl struct
ints_syl = map((x, y) -> (x, x + y), offs_syl, durs_syl)
syls = MarkedInterval.(ints_syl, labels_syl)

# Get motif info
motif_i = find_subseq(motif, labels_syl)
motif_ons, motif_ofs, motif_durs = motif_info(offs_syl, durs_syl, motif_len, motif_i)
n_motif = length(motif_i)
p = sortperm(motif_durs, rev = true)

prev = similar(p)
prev[p] = 1:n_motif

# Define bounds for display
med_dur = median(motif_durs)
post = med_dur + 0.2
pre = 0.15

xdur = pre + post

# Get syllables that will be displayed
motif_sylls, other_sylls, int_mask = raster_syll_data(
    syls, motif_i, motif_len, pre, post
);
other_syl_ints = map(x->map(y->y.interval, x), other_sylls)
motif_syl_ints = map(x->map(y->y.interval, x), motif_sylls)

aligned_m_syl_ints = align_intervals(motif_syl_ints, motif_ons)
aligned_o_syl_ints = align_intervals(other_syl_ints, motif_ons)

spk_dur = 0.003
spk_wv_support = ceil(Int, spk_dur * fs)
half_basis = div(spk_wv_support, 2)

# Data for this unit
spk_idx = 8

motif_spks = motif_events(spks[spk_idx], int_mask)

aligned_spks = align_events(motif_spks, motif_ons)

example_rep = argmin([ifelse(isempty(motif_spks[p[i]]), Inf, abs(med_dur - motif_durs[p[i]])) for i in 1:n_motif])

orig_ex_rep = p[example_rep]

motif_idx_b = t_to_ndx(int_mask[orig_ex_rep][1], fs)
motif_idx_e = t_to_ndx(int_mask[orig_ex_rep][2], fs)
song_clip = mic_arr[motif_idx_b:motif_idx_e]

spk_rec = neur_map[recording_ids[spk_idx]]
rec_clip = spk_rec[motif_idx_b:motif_idx_e]
spk_clip_filt, _, _ = filtfilt_mmap(rec_clip, fs)

whitened_spk_clips = map(
    t -> spike_clip_fixed(
        neur_whitened_map[recording_ids[spk_idx]],
        round(Int, t * fs),
        half_basis
    ),
    spks[spk_idx]
)

max_gap = 3
bout_intervals = join_intervals(ints_syl, max_gap)
silence_intervals = interval_compliments(0, mic_rec_dur, bout_intervals, max_gap)

bout_durs = map(x -> x[2] - x[1], bout_intervals)
silence_durs = map(x -> x[2] - x[1], silence_intervals)


iei_binsize = 5
iei_max = 200
iei_bins = 0:iei_binsize:iei_max

bout_spks = map(t -> mask_events(spks[spk_idx], t[1], t[2]), bout_intervals)
silence_spks = map(t -> mask_events(spks[spk_idx], t[1], t[2]), silence_intervals)

ac_binsize = 0.005
ac_maxdiff = 0.3
bar_width = 0.9
ac_sing, ac_c_sing = acorr_discrete_normed(
    bout_spks, bout_durs, binsize = ac_binsize, maxdiff = ac_maxdiff
)
ac_silence, ac_c_silence = acorr_discrete_normed(
    silence_spks, silence_durs, binsize = ac_binsize, maxdiff = ac_maxdiff
)

a_bout_spks = map((s, b) -> s .- b[1], bout_spks, bout_intervals)
a_sil_spks = a_bout_spks = map((s, b) -> s .- b[1], silence_spks, silence_intervals)

psd_sing, f_sing = point_psd(a_bout_spks, bout_durs, fs)
psd_sil, f_sil = point_psd(a_sil_spks, silence_durs, fs)

bout_fr = sum(length, bout_spks) / sum(bout_durs)
sil_fr = sum(length, silence_spks) / sum(silence_durs)
global_fr = length(spks[spk_idx]) / durs_recs[spk_idx]

silence_ieis = sort!(cat(event_intervals.(silence_spks)..., dims = 1))
singing_ieis = sort!(cat(event_intervals.(bout_spks)..., dims = 1))

ks_t = ApproximateTwoSampleKSTest(singing_ieis, silence_ieis)

bin_cost = function (b)
    normed_weights, edges, raw_counts = hist_density_reps(
        aligned_spks, -pre, post, binsize = b
    )
    c = (2 * mean(raw_counts) - var(raw_counts)) / (133 * b) ^ 2
end

opt = optimize(bin_cost, 0.001, post)

binsize = opt.minimizer
normed_weights, edges, raw_counts = hist_density_reps(
    aligned_spks, -pre, post, binsize = binsize
)

mn_t = MultinomialLRT(raw_counts)


spk_wavs = make_lc_coords((-half_basis:1:half_basis)/30, whitened_spk_clips)

loc_string = @sprintf "(%.1fµm, %.1fµm)" x_ch[spk_idx] y_ch[spk_idx]
fr_format = x -> @sprintf "%.1fHz" x
time_format = x -> @sprintf "%.2fs" x
depth_format = x -> @sprintf "%.0fµm" x
pval_show = x -> string(StatsBase.PValue(pvalue(x)))

# Make figure
rc = PyPlot.matplotlib[:rc]
rc("axes", titlesize= 10)
rc("axes", labelsize = 8)
rc("xtick", labelsize = 6)
rc("ytick", labelsize = 6)
rc("legend", fontsize = 6)

f = figure(figsize = (8.5, 11), dpi = 120)

gs_left = PyPlot.matplotlib[:gridspec][:GridSpec](
    4, 2, left = 0.05, right = 0.48
)

meta_ax = f[:add_subplot](gs_left[:__getitem__]((0, pybuiltin(:slice)(0, 2))))
pos_ax = f[:add_subplot](gs_left[2, 1])
overlay_ax = f[:add_subplot](gs_left[2, 2])
iei_ax = f[:add_subplot](gs_left[3, 1])
qq_ax = f[:add_subplot](gs_left[3, 2])
acorr_ax = f[:add_subplot](gs_left[4, 1])
psd_ax = f[:add_subplot](gs_left[4, 2])

gs_right = PyPlot.matplotlib[:gridspec][:GridSpec](
    9, 1, left = 0.55, right = 0.98
)

spax = f[:add_subplot](gs_right[1])
tsax = f[:add_subplot](gs_right[2])
rax = f[:add_subplot](gs_right[:__getitem__](pybuiltin(:slice)(2, 5)))
pax = f[:add_subplot](gs_right[6])

spax[:get_shared_x_axes]()[:join](spax, tsax, rax, pax)
rax[:set_xticklabels]([])

gl_ax = Axis{MPL}.([spax, tsax, rax, pax])

raster_plot(
    rax,
    aligned_spks[p],
    [aligned_m_syl_ints[p], aligned_o_syl_ints[p]],
    pre,
    post
)
th = rax[:text](post + 0.007 , example_rep, "*", ha = "left", va = "center")
rax[:set_ylabel]("Motif No.")
rax[:set_title]("Song locking")

centers = edges[1:end-1] .+ binsize ./ 2

distplot(
    centers,
    bins = edges,
    color = "tab:gray",
    kde = false,
    ax = pax,
    hist_kws = Dict("weights" => normed_weights)
)
pax[:set_ylabel]("Firing rate (Hz)")
pax[:set_xlabel]("Time from motif onset (s)")
pax[:spines]["right"][:set_visible](false)
pax[:spines]["top"][:set_visible](false)

rs = resizeable_spectrogram(
    gl_ax[1],
    song_clip .- mean(song_clip),
    fs,
    -pre,
    clim = clim,
    frange = frange,
    listen_ax = gl_ax[2:end]
);

spax[:set_title]("Example recording", y = 1.3)
spax[:axis]("off")

left_scale_x = -pre - 0.05 * xdur

sb_sp = matplotlib_scalebar(
    spax, 4000, "4kHz", horizontal = false,
    loc = "lower right", axes_pos = [0, 0], sep = 2,
    textprops = Dict("fontsize" => 6)
)

# Syllable labels
clipped_o = clip_int.(aligned_o_syl_ints[orig_ex_rep], Ref((-pre, post)))

pcm = make_patch_collection(
    aligned_m_syl_ints[orig_ex_rep];
    height = 1000,
    ycenter = 13500,
    facecolor =  "#9ecae1",
    clip_on = false
)

pco = make_patch_collection(
    clipped_o;
    height = 1000,
    ycenter = 13500,
    facecolor = "#deebf7",
    clip_on = false
)

spax[:add_collection](pcm)
spax[:add_collection](pco)

mth = add_labels(spax, mean.(aligned_m_syl_ints[orig_ex_rep]), motif, 14000)

oth = add_labels(
    spax,
    mean.(clipped_o),
    getfield.(other_sylls[orig_ex_rep], :mark),
    14000
)

rts = downsamp_patch(
    gl_ax[2], spk_clip_filt, fs, -pre,
    color = "k",
    listen_ax = gl_ax[[1, 3, 4]],
    linewidth = 0.5
)

tsax[:axis]("off")

sb_v = matplotlib_scalebar(
    tsax, 100, "100µV", horizontal = false,
    loc = "center right", axes_pos = [0, 0.5], sep = 2,
    textprops = Dict("fontsize" => 6)
)

sb_ms = matplotlib_scalebar(
    tsax, 0.05, "50ms", textfirst = false,
    loc = "lower right", axes_pos = [0.9, 1],
    textprops = Dict("fontsize" => 6)
)

ac_s_h = acorr_ax[:bar](
    ac_c_sing * 1000, ac_sing, 1000 * bar_width * ac_binsize, alpha = 0.4
)
ac_ns_h = acorr_ax[:bar](
    ac_c_silence * 1000, ac_silence, 1000 * bar_width * ac_binsize, alpha = 0.4
)

acorr_ax[:axhline](color = "k")
acorr_ax[:set_xlim]([0, 1000 * (ac_c_silence[end] + ac_binsize / 2)])
# acorr_ax[:legend]([ac_s_h, ac_ns_h], ["singing", "silence"])
acorr_ax[:set_ylabel]("correlation")
acorr_ax[:set_xlabel]("lag (ms)")
acorr_ax[:set_title]("Spike autocorrelation")

h_psd_sing = psd_ax[:plot](f_sing, psd_sing * 10^9, alpha = 0.5)[1]
h_psd_sil = psd_ax[:plot](f_sil, psd_sil * 10^9, alpha = 0.5)[1]
psd_ax[:set_xlim]([1, 100])
psd_ax[:set_ylim]([0, global_fr * 1.25])
psd_ax[:set_xscale]("log")
psd_ax[:grid](true, which = "both", axis = "x")
psd_ax[:set_xlabel]("Frequency (Hz)")
psd_ax[:set_ylabel](raw"$x10^{-9}spike^2/Hz$")
psd_ax[:set_title]("Spike PSD")

distplot(
    singing_ieis .* 1000,
    bins = iei_bins,
    norm_hist = true,
    kde = false,
    ax = iei_ax
)
distplot(
    silence_ieis .* 1000,
    bins = iei_bins,
    norm_hist = true,
    kde = false,
    ax = iei_ax
)
iei_ax[:set_xlim]([0, iei_max])
iei_ax[:legend](["singing", "silence"])
iei_ax[:set_title]("Interspike intervals")
iei_ax[:set_xlabel]("ISI (ms)")
iei_ax[:set_ylabel]("Probability density")
iei_ax[:tick_params](
    axis = "y", which = "both", left = false, right = false, labelleft = false
)

qqy, qqx = destruct(empirical_qq(singing_ieis .* 1000, silence_ieis .* 1000))
iei_range = extrema_red(extrema.([singing_ieis .* 1000, silence_ieis .* 1000]))
qq_s = qq_ax[:scatter](qqx, qqy, 10, marker = ".", alpha = 0.3, color = "k")
qq_l = qq_ax[:plot](iei_range, iei_range, ":k", linewidth = 1)[1]
qq_ax[:set_xscale]("log")
qq_ax[:set_yscale]("log")
qq_ax[:set_xlim]([iei_range...])
qq_ax[:set_ylim]([iei_range...])
qq_ax[:set_xlabel]("ISI in silence (ms)")
qq_ax[:set_ylabel]("ISI in song (ms)")
qq_ax[:set_title]("Q-Q for ISIs")

lc = PyPlot.matplotlib[:collections][:LineCollection](
    spk_wavs, color = (0,0,0,0.3), linewidths = 0.3
)

overlay_ax[:set_xlim]([-half_basis / 30, half_basis / 30])
overlay_ax[:add_collection](lc)
overlay_ax[:autoscale](axis = "y")
overlay_ax[:set_xlabel]("Time (ms)")
overlay_ax[:set_ylabel]("Std. from baseline")
overlay_ax[:set_title]("Waveforms")

c_this_elec = PyPlot.matplotlib[:patches][:Circle](
    (x_ch[spk_idx], y_ch[spk_idx]), NeuroPlots.PI_PITCH / 2,
    edgecolor = "none"
)
pos_ax[:add_patch](c_this_elec)
pc_elec = electrode_circles(facecolor = "none", edgecolor = "k", linewidth = 1)
pos_ax[:add_collection](pc_elec)
pos_ax[:autoscale]()
pos_ax[:set_aspect]("equal", "datalim")
pos_ax[:set_title]("Electrode")
pos_ax[:set_xlabel]("µm")
pos_ax[:set_ylabel]("µm")

T = meta_ax[:table](
    cellText = [
        ["Bird", "DPH", "Song rec_ID"],
        ["7358", "68", string(song_rid)],

        ["Channel", "rec_ID", "Location"],
        [string(chnos[spk_idx]), string(recording_ids[spk_idx]), loc_string],

        ["Rec. duration", "Singing dur.", "Silence dur."],
        [
            time_format(durs_recs[spk_idx]),
            time_format(sum(bout_durs)),
            time_format(sum(silence_durs))
        ],

        ["depth", "Motif", "# of motifs"],
        ["", string(motif...), string(length(motif_i))],

        ["FR global", "FR song", "FR silence"],
        [fr_format(global_fr), fr_format(bout_fr), fr_format(sil_fr)],

        ["p-val ISI", "p-val locking", ""],
        [pval_show(ks_t), pval_show(mn_t), ""]
    ],
    loc = "center"
)
meta_ax[:axis]("off")


gs_right[:update](left = 0.55, right = 0.95, bottom = 0.08, top = 0.90, hspace = 0.5)

gs_left[:tight_layout](f, rect = [0.05, 0.05, 0.5, 0.95])

f[:savefig]("summary.pdf", transparent = true)
