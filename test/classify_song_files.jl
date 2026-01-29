using Revise,
    LibPQ,
    OEDiscovery,
    PyCall,
    PyPlot,
    TSConditioning,
    SpikeAnalysis,
    AcqGuiTools,
    Dates,
    GLFileCache,
    Destruct,
    GLUtilities,
    LIBSVM

const conn_str = "postgresql://galen@localhost:5433/galen";
const conn = LibPQ.Connection(conn_str);
const sr = 40000
const subs_searchdir = "/home/glynch/Documents/Data/Screening/7406/2018-04-05"

q_str = """
SELECT DISTINCT bird_id, lower(event_time_range)::date AS rec_date
FROM contexts_birds
NATURAL INNER JOIN events
NATURAL INNER JOIN event_types
NATURAL INNER JOIN events_labels
NATURAL INNER JOIN labels
WHERE type_name = 'syllable'
AND label_name = 'juvenile';
""";

nt = execute_fetch(conn, q_str, NamedTuple, true);
anno_dates = Date.(nt.rec_date);
b_song = make_bandpass(40000, 1000, 6000, 151);

function song_predictor_files(files::AbstractVector{<:AbstractString}, b_song, sr)
    nf = length(files)
    mean_song_i = Vector{Float64}(undef, nf)
    rhythm_pow = similar(mean_song_i)
    for i = 1:nf
        agd = AcqGuiData(files[i], Float64)
        mean_song_i[i], rhythm_pow[i] = score_array_for_song(b_song, agd.data, sr)
    end
    mean_song_i, rhythm_pow
end

function song_predictor_labels(
    conn::LibPQ.Connection,
    bird_id::Integer,
    anno_date::Date,
    b_song::AbstractVector,
    sr::Number,
)
    nt = get_ag_files_and_song_status(conn, bird_id, anno_date)
    dbf = dbfiles(nt)
    nf = length(dbf)
    mean_song_i = Vector{Float64}(undef, nf)
    rhythm_pow = similar(mean_song_i)
    for i = 1:nf
        fpath = find_db_file(dbf[i])
        agd = AcqGuiData(fpath, Float64)
        mean_song_i[i], rhythm_pow[i] = score_array_for_song(b_song, agd.data, sr)
    end
    mean_song_i, rhythm_pow, nt.has_song
end

all_song_i, all_rhythm_pow, all_has_song = destruct(
    map((id, d) -> song_predictor_labels(conn, id, d, b_song, sr), nt.bird_id, anno_dates),
)

flat_song_i = reduce(vcat, all_song_i)
flat_rhythm_pow = reduce(vcat, all_rhythm_pow)
flat_has_song = reduce(vcat, all_has_song)

identifer = 192269

og = oe_ag_offsets_and_duration(conn, identifier)

subs_files = readdir(subs_searchdir);
subs_paths = joinpath.(Ref(subs_searchdir), subs_files);
m = only_matches(AG_REGEX, subs_files);
fnos = parse.(Int, getindex.(getfield.(m, :captures), 2));
lookup = Dict(no => subs_paths[i] for (i, no) in enumerate(fnos));

song_f = getindex.(Ref(lookup), 689:694)
non_song_f = getindex.(Ref(lookup), 589:688)

subs_song_i, subs_song_rhythm_pow = song_predictor_files(song_f, b_song, sr)
subs_non_song_i, subs_non_song_rhythm_pow = song_predictor_files(non_song_f, b_song, sr)

n_juv = length(flat_song_i)
n_subs = length(subs_song_i)
n_subs_cage = length(subs_non_song_i)
tot = n_juv + n_subs + n_subs_cage
feats = Matrix{Float64}(undef, 2, tot);
labels = Vector{Symbol}(undef, tot);

feats[1, 1:n_juv] = flat_song_i;
feats[2, 1:n_juv] = flat_rhythm_pow;
feats[1, (n_juv+1):(n_juv+n_subs)] = subs_song_i;
feats[2, (n_juv+1):(n_juv+n_subs)] = subs_song_rhythm_pow;
feats[1, (n_juv+n_subs+1):(n_juv+n_subs+n_subs_cage)] = subs_non_song_i;
feats[2, (n_juv+n_subs+1):(n_juv+n_subs+n_subs_cage)] = subs_non_song_rhythm_pow;

labels[1:n_juv] = map(x -> ifelse(x, :song, :non_song), flat_has_song)
labels[(n_juv+1):(n_juv+n_subs)] .= :sub_song
labels[(n_juv+n_subs+1):(n_juv+n_subs+n_subs_cage)] .= :non_song

subs_mask = labels .== :sub_song
cage_noise_mask = labels .== :non_song
song_mask = .! cage_noise_mask
subs_data_mask = subs_mask .| cage_noise_mask
n_song = sum(song_mask)
n_nonsong = tot - n_song

subs_feats = feats[:, subs_data_mask]
subs_labels = labels[subs_data_mask]
subs_model = svmtrain(subs_feats, subs_labels)

pta = (-66.6, 104)
ptb = (-47.7, 8.08)
disc_slope = -5.075132275132278
disc_intercept = -234.00380952380965

line_pt_dist(x, y) = - (disc_slope * x - y + disc_intercept) / sqrt(disc_slope ^ 2 + 1)

projs = Vector{Float64}(undef, tot)
for i = 1:tot
    projs[i] = line_pt_dist(feats[1, i], feats[2, i])
end

d_vals = -25:0.1:25
roc_pts = Matrix{Float64}(undef, 2, length(d_vals))
for (i, dp) in enumerate(d_vals)
    class = projs .>= dp
    roc_pts[1, i] = sum(class .& song_mask) / n_song
    roc_pts[2, i] = sum(class .& cage_noise_mask) / n_nonsong
end

ld = fit(LinearDiscriminant, feats[:, song_mask], feats[:, cage_noise_mask])

ld_projs = Vector{Float64}(undef, tot)
for i = 1:tot
    ld_projs[i] = evaluate(ld, feats[:, i])
end

roc_lda = Matrix{Float64}(undef, 2, length(d_vals))
for (i, dp) in enumerate(d_vals)
    class = ld_projs .>= dp
    roc_lda[1, i] = sum(class .& song_mask) / n_song
    roc_lda[2, i] = sum(class .& cage_noise_mask) / n_nonsong
end
