using
    Revise,
    LibPQ,
    PyCall,
    SpikeAnalysis,
    PyPlot,
    Seaborn,
    Statistics,
    OEDiscovery,
    Destruct


const conn_str =  "postgresql://galen@localhost:5433/galen"
const conn = LibPQ.Connection(conn_str)

const song_rid = 192243

(
    recording_ids,
    chnos,
    spk_type,
    labels_spk,
    durs_recs,
    spks,
    x_ch,
    y_ch,
    z_ch
) = get_spikes(conn, song_rid)

spk_idx = 1

xc, c = acorr_discrete_normed(spks[spk_idx], durs_recs[spk_idx])
xc, c = xcorr_discrete_normed(spks[spk_idx], spks[spk_idx], durs_recs[spk_idx])
