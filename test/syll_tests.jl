using Revise,
    LibPQ,
    DbCache,
    DataFrames,
    OpenEphysLoader,
    OEUtilities,
    AcqGuiTools,
    GLFileCache,
    GLUtilities,
    Destruct


const connstr = "postgresql://galen@localhost:5433/galen"
conn = LibPQ.Connection(connstr)
const storagedir = "/home/glynch/Documents/Data/temp_storage/"
const cachedir = "/home/glynch/Documents/Data/temp_storage/caching"
context = :laptop

lookaside = build_lookaside(storagedir)

oe_rec_id = 192243

nt = fetch!(
    NamedTuple,
    execute(
        conn,
        """
WITH event_fileinfo AS (
SELECT event_id,
    EXTRACT(EPOCH FROM
        lower(event_time_range) - lower(recording_time_range)
    ) AS offset,
    EXTRACT(EPOCH FROM
        upper(event_time_range) - lower(event_time_range)
    ) AS duration,
    recording_id,
     file_id,
     host_name,
     path_name,
     file_name
FROM events e
NATURAL INNER JOIN recordings_events re
NATURAL INNER JOIN recordings r
NATURAL INNER JOIN files_recordings
NATURAL INNER JOIN files
NATURAL INNER JOIN hosts
NATURAL INNER JOIN paths
)
SELECT oe.file_id AS oe_file_id,
        oe.host_name AS oe_host_name,
        oe.path_name AS oe_path_name,
        oe.file_name AS oe_file_name,
        oe.duration AS oe_duration,
        oe.offset AS oe_offset,
        ae.file_id AS ae_file_id,
    ae.host_name AS ae_host_name,
    ae.path_name AS ae_path_name,
    ae.file_name AS ae_file_name,
    ae.duration AS ae_duration,
    ae.offset AS ae_offset
FROM derived_events de
INNER JOIN event_fileinfo oe ON (oe.event_id = de.derived_event_id)
INNER JOIN event_fileinfo ae ON (ae.event_id = de.original_event_id)
""",
        not_null = true,
    ),
)


function get_syllpairs(oef, agf, oe_offset, oe_dur, ag_offset, ag_dur)
    oesyll = get_syll(SampleArray, oef, oe_offset, oe_dur)
    agsyll = get_syll(AcqGuiData, agf, ag_offset, ag_dur)
    oesyll, agsyll
end

function get_syll(::Type{AcqGuiData}, f::AbstractString, offset, duration)
    open(f, "r") do io
        A = AcqGuiData(io, Float64)
        fs = A.fs
        get_syll(A.data, fs, offset, duration)
    end
end

function get_syll(::Type{SampleArray}, f::AbstractString, offset, duration)
    open(f, "r") do io
        A = SampleArray(Float32, io, false)
        fs = A.contfile.header.samplerate
        get_syll(A, fs, offset, duration)
    end
end

function get_syll(A::AbstractArray, fs, offset, duration)
    na = length(A)
    i_b = t_to_ndx(offset, fs)
    i_e = i_b + t_to_ndx(duration, fs)
    A[clip_ndx(i_b, na):clip_ndx(i_e, na)]
end

oe_dbfs = dbfiles(nt.oe_file_id, nt.oe_host_name, nt.oe_path_name, nt.oe_file_name)
ag_dbfs = dbfiles(nt.ae_file_id, nt.ae_host_name, nt.ae_path_name, nt.ae_file_name)
oefs = find_db_file.(oe_dbfs)
agfs = find_db_file.(ag_dbfs)

oesylls, agsylls = destruct(
    get_syllpairs.(oefs, agfs, nt.oe_offset, nt.oe_duration, nt.ae_offset, nt.ae_duration),
)
fig, pyax = subplots(2, 1, sharex = true, sharey = true)
ax = Axis{MPL}.(pyax)
pair_no = 3
outsa = resizeable_spectrogram(ax[1], oesylls[pair_no], 30000);
outsb = resizeable_spectrogram(ax[2], agsylls[pair_no], 40000);
