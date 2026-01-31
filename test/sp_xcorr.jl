using Compat,
    GlDbPlotting,
    GLFileCache,
    LibPQ,
    OEUtilities,
    TSConditioning,
    DSP,
    OpenEphysLoader,
    SignalPlots,
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
    Combinatorics

const conn_str = "postgresql://galen@localhost:5433/galen"
const sql_q_str = """
WITH seed_rid AS (
     SELECT \$1::integer AS recording_id
), neural_chans AS (
   SELECT overlapping_recording_id AS recording_id
   FROM seed_rid sr
   INNER JOIN recording_groups rg ON (rg.seed_recording_id = sr.recording_id)
   INNER JOIN timeseries_neural tn ON (
         tn.recording_id = rg.overlapping_recording_id
   )
), neural_events AS (
   SELECT recording_id,
          type_name,
          label_name,
          EXTRACT(
            EPOCH FROM (
                  lower(event_time_range) - lower(recording_time_range) +
                  (upper(event_time_range) - lower(event_time_range)) / 2
            )
          ) AS seconds_from_start
   FROM neural_chans
   INNER JOIN recordings r USING (recording_id)
   INNER JOIN recordings_events USING (recording_id)
   INNER JOIN events USING (event_id)
   INNER JOIN event_types USING (type_id)
   INNER JOIN events_labels USING (event_id)
   INNER JOIN labels USING (label_id)
), spikes AS (
   SELECT recording_id,
          type_name,
          label_name,
          array_agg(seconds_from_start ORDER BY seconds_from_start) AS spikes
   FROM neural_events
   GROUP BY recording_id, type_name, label_name
)
SELECT type_name,
       label_name,
       recording_id,
       channel_number,
       point_x,
       point_y,
       point_z,
       spikes,
       EXTRACT(
            EPOCH FROM (
                upper(recording_time_range) - lower(recording_time_range)
            )
          ) AS recording_dur
FROM spikes
INNER JOIN recordings_with_spatial USING (recording_id)
INNER JOIN timeseries_neural USING (recording_id)
ORDER BY type_name, label_name, recording_id, point_x, point_y;
"""

const conn = LibPQ.Connection(conn_str)

const spike_stmt = prepare(conn, sql_q_str)

const song_rid = 192243

const spkdf = fetch!(DataFrame, execute(spike_stmt, [song_rid]))

const ch_of_interest = [6, 26, 28]

mask =
    (spkdf[:type_name] .== "spike-good") .&
    map(x->x in ch_of_interest, spkdf[:channel_number])

gooddf = spkdf[mask, :]
xs = disallowmissing(gooddf[:point_x])
ys = disallowmissing(gooddf[:point_y])
vecs = [[x, y] for (x, y) in zip(xs, ys)]

spv = disallowmissing(gooddf[:spikes])

to_displ_vec(df, idx) = [df[idx, :point_x], df[idx, :point_y]]

maxlag = 1
lagmask = abs.(centers) .<= maxlag;

fig, ax = subplots(3, 1, sharex = true, sharey = true)

function plotpair(idx, ia, ib)
    xc, centers = xcorr_discrete_normed(spv[ia], spv[ib], dur, binsize = 0.01)
    d = euclidean(vecs[ia], vecs[ib])
    namea = gooddf[ia, :channel_number]
    nameb = gooddf[ib, :channel_number]
    ax[idx][:plot](centers[lagmask], xc[lagmask], label = "$namea - $nameb")
    ax[idx][:set_title]("$d um")
    ax[idx][:legend]()
end

for (i, (a, b)) in enumerate(combinations(1:size(gooddf, 1), 2))
    plotpair(i, a, b)
end

xc, centers = xcorr_discrete_normed(spv[1], spv[3], dur, binsize = 0.01)
subplot(3, 1, 2)
plot(centers[lagmask], xc[lagmask])

xc, centers = xcorr_discrete_normed(spv[2], spv[3], dur, binsize = 0.01)
subplot(3, 1, 2)
plot(centers[lagmask], xc[lagmask])

muachans = 12:22
muamask =
    (spkdf[:type_name] .== "spike-mua") .& map(x->x in muachans, spkdf[:channel_number])

muadf = spkdf[muamask, :]

nmua = size(muadf, 1)

idxpairs = collect(combinations(1:nmua, 2))
npair = length(idxpairs)

mua_xcorrs = Vector{NTuple{3,Float64}}(undef, npair)
xs = disallowmissing(muadf[:point_x])
ys = disallowmissing(muadf[:point_y])
vecs = [[x, y] for (x, y) in zip(xs, ys)]

for i = 1:npair
    ia, ib = idxpairs[i]
    d = euclidean(vecs[ia], vecs[ib])
    xc, centers =
        xcorr_discrete_normed(muadf[ia, :spikes], muadf[ib, :spikes], dur, binsize = 0.01)
    clxc = xc[lagmask]
    clc = centers[lagmask]
    ibest = argmax(abs.(clxc))
    bestlag = clc[ibest]
    bestxcorr = clxc[ibest]
    mua_xcorrs[i] = (d, bestlag, bestxcorr)
end

bigmask = getindex.(mua_xcorrs, 3) .> 0.1
followups = idxpairs[bigmask]

nfol = length(followups)

fig, ax = subplots(nfol, 1, sharex = true, sharey = true)
for i = 1:nfol
    ia, ib = followups[i]
    xc, centers =
        xcorr_discrete_normed(muadf[ia, :spikes], muadf[ib, :spikes], dur, binsize = 0.01)
    d = euclidean(to_displ_vec(muadf, ia), to_displ_vec(muadf, ib))
    namea = muadf[ia, :channel_number]
    nameb = muadf[ib, :channel_number]
    ax[i][:plot](centers[lagmask], xc[lagmask], label = "$namea - $nameb")
    ax[i][:set_title]("$d um")
    ax[i][:legend]()
end
