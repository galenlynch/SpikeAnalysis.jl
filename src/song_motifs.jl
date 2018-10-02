struct RasterTrigSylls{D, M<:MarkedInterval{D, <:Any}}
    triggered::Vector{M}
    other::Vector{M}
    motif_interval::NTuple{2, D}
    expanded_interval::NTuple{2, D}
end

struct TrigSet{D, R<:RasterTrigSylls{D, <:Any}}
    trigs::Vector{R}
    trig_durs::Vector{D}
    median_trig_dur::D
    pre::D
    post::D
end

function raster_syll_data(
    sylls::AbstractVector{<:MarkedInterval},
    motif_i::AbstractArray{<:Integer},
    motif_len::Integer,
    pre = 0.05, post = 0.55;
)
    n_motif = length(motif_i)
    motif_sylls, other_sylls, int_mask = destruct(
        map(
            i -> select_motif_sylls(sylls, i, i + motif_len - 1, pre, post),
            motif_i
        )
    )
end

function motif_events(
    event_set::AbstractVector,
    int_masks::AbstractVector{<:NTuple{2, <:Number}}
)
    map(
        int_motif -> mask_events(event_set, int_motif[1], int_motif[2]),
        int_masks
    )
end

# sylls must be sorted
function select_motif_sylls(
    sylls::AbstractVector{M},
    motif_b::Integer,
    motif_e::Integer,
    pre::Real,
    post::Real,
    nsyll::Integer = length(sylls)
) where M<:MarkedInterval
    motif_sylls = view(sylls, motif_b:motif_e)
    onset = bounds(motif_sylls[1].interval)[1]
    int_mask = (onset - pre, onset + post)

    # The following is a hack to get around the behavior of searchsortedfirst
    # Basically the 'by' argument applies to the collection as well as the
    # element being compared.
    # The goal is to compare  the OFFSET of syllables to the ONSET of the
    # the motif range, so the 'search_interval' term below has its interval
    # swapped to get the right answer for the searches
    search_interval = (interval = NakedInterval(int_mask[2], int_mask[1]),)
    prior_syll_idx = searchsortedfirst(
        view(sylls, 1:(motif_b - 1)),
        search_interval,
        by = x -> bounds(x.interval)[2]
    )
    n_ps = searchsortedlast(
        view(sylls, (motif_e + 1):nsyll),
        search_interval,
        by = x -> bounds(x.interval)[1]
    )
    if prior_syll_idx < motif_b
        n_pr = n_ndx(prior_syll_idx, motif_b - 1)
    else
        n_pr = 0
    end
    other_sylls = Vector{M}(undef, n_pr + n_ps)
    other_sylls[1:n_pr] = sylls[(motif_b - n_pr):(motif_b - 1)]
    other_sylls[(1 + n_pr):end] = sylls[(motif_e + 1):ndx_offset(motif_e + 1, n_ps)]
    motif_sylls, other_sylls, int_mask
end

function align_events(motif_events, motif_onsets)
    map((x, o) -> x .- o, motif_events, motif_onsets)
end

function align_intervals(intervals, onsets)
    map(
        (ints, o) -> map(
            t -> (t[1] - o, t[2] - o),
            ints
        ),
        intervals,
        onsets
    )
end

function motif_info(offs_syl, durs_syl, n_motif, motif_i)
    offs_i = motif_i .+ n_motif .- 1
    motif_ons = offs_syl[motif_i]
    motif_ofs = offs_syl[offs_i] .+ durs_syl[offs_i]
    motif_durs = motif_ofs .- motif_ons
    motif_ons, motif_ofs, motif_durs
end

function find_motifs(
    motif::AbstractVector{<:AbstractString},
    labels_syl,
    offs_syl,
    durs_syl,
    motif_len
)
    # Get motif info
    motif_i = find_subseq(motif, labels_syl)
    motif_ons, motif_offs, motif_durs = motif_info(
        offs_syl, durs_syl, motif_len, motif_i
    )
    p = sortperm(motif_durs, rev = true)
    return motif_i[p], motif_ons[p], motif_offs[p], motif_durs[p]
end

function call_bouts(ints_syl, mic_rec_dur, max_gap = 3)
    bout_intervals = join_intervals(ints_syl, max_gap)
    silence_intervals = interval_compliments(
        0, mic_rec_dur, bout_intervals, max_gap
    )

    bout_durs = map(x -> x[2] - x[1], bout_intervals)
    silence_durs = map(x -> x[2] - x[1], silence_intervals)
    return bout_intervals, silence_intervals, bout_durs, silence_durs
end

function trig_data(
    trig_seq::AbstractVector{<:AbstractString},
    labels_syl::AbstractVector{<:AbstractString},
    offs_syl::AbstractVector{<:Real},
    durs_syl::AbstractVector{<:Real},
    syls::AbstractVector{<:MarkedInterval},
    pre::Real,
    post_expand::Union{Real, Nothing} = nothing;
    post::Union{Nothing, Real} = nothing
)
    trig_len = length(trig_seq)
    motif_i, motif_ons, motif_offs, motif_durs = find_motifs(
        trig_seq, labels_syl, offs_syl, durs_syl, trig_len
    )

    median_motif_dur = median(motif_durs)
    if post == nothing
        post_expand == nothing && error("post or post_expand must be specified")
        post = median_motif_dur + post_expand
    end

    # Get syllables that will be displayed
    motif_sylls, other_sylls, int_masks = raster_syll_data(
        syls, motif_i, trig_len, pre, post
    )

    trigs = map(
        (msyl, osyl, mb, me, mask_int) -> RasterTrigSylls(
            collect(msyl), collect(osyl), (mb, me), mask_int
        ),
        motif_sylls, other_sylls, motif_ons, motif_offs, int_masks
    )

    TrigSet(trigs, motif_durs, median_motif_dur, pre, post)
end

function group_align_events(event_set::AbstractVector, trigs::TrigSet)
    map(rt -> group_align_events(event_set, rt), trigs.trigs)
end
function group_align_events(event_set, trig::RasterTrigSylls)
    mask_events(event_set, trig.expanded_interval...) .-
        bounds(trig.triggered[1].interval)[1]
end

function motif_events(event_set::AbstractVector, trigs::TrigSet)
    map(rt -> motif_events(event_set, rt), trigs.trigs)
end

function motif_events(event_set::AbstractVector, trig::RasterTrigSylls)
    mask_events(event_set, trig.expanded_interval...)
end
function align_trigs(raw::TrigSet)
    aligned_trigs = similar(raw.trigs)
    for (i, t) in enumerate(raw.trigs)
        aligned_trigs[i] = align_trigs(t)
    end
    TrigSet(
        aligned_trigs,
        raw.trig_durs,
        raw.median_trig_dur,
        raw.pre,
        raw.post
    )
end

function align_trigs(raw::RasterTrigSylls)
    onset = bounds(raw.triggered[1].interval)[1]
    RasterTrigSylls(
        align_intervals(raw.triggered, onset),
        align_intervals(raw.other, onset),
        (raw.motif_interval[1] - onset, raw.motif_interval[2] - onset),
        (raw.expanded_interval[1] - onset, raw.expanded_interval[2] - onset)
    )
end

function align_intervals(raw::AbstractVector{<:MarkedInterval}, onset)
    aligned = similar(raw)
    for (i, int) in enumerate(raw)
        aligned[i] = MarkedInterval(
            (bounds(int.interval)[1] - onset, bounds(int.interval)[2] - onset),
            int.mark
        )
    end
    aligned
end

function clip_trigs(raw::TrigSet)
    clipped_trigs = similar(raw.trigs)
    for (i, t) in enumerate(raw.trigs)
        clipped_trigs[i] = clip_trigs(t)
    end
    TrigSet(
        clipped_trigs,
        raw.trig_durs,
        raw.median_trig_dur,
        raw.pre,
        raw.post
    )
end

function clip_trigs(raw::RasterTrigSylls)
    RasterTrigSylls(
        raw.triggered,
        clip_trigs(raw.other, raw.expanded_interval),
        raw.motif_interval,
        raw.expanded_interval
    )
end

function clip_trigs(raw::AbstractVector{<:MarkedInterval}, clip_bnds)
    clipped = similar(raw)
    for (i, int) in enumerate(raw)
        clipped[i] = MarkedInterval(
            clip_int(bounds(int.interval), clip_bnds), int.mark
        )
    end
    clipped
end

function align_events(motif_events::AbstractVector{<:AbstractVector}, rt::TrigSet)
    map(align_events, motif_events, rt.trigs)
end
function align_events(motif_events::AbstractVector{<:Number}, rt::RasterTrigSylls)
    motif_events .- rt.triggered[1][1]
end

function add_pres(
    rec::NakedInterval{E},
    sylls::AbstractVector{M},
    trigsylls::AbstractVector{<:String};
    pre_syll::Real = 0.03,
) where {E, M<:MarkedInterval{E, <:Any}}
    n_syll = length(sylls)
    expanded = Vector{
        IntervalSet{E, Tuple{RelativeInterval{E, M, M}, M}}
    }(undef, n_syll)
    points = Vector{NakedPoints{E, NakedInterval{E}, Vector{E}}}(undef, n_syll)
    trigno = 0
    lastend = bounds(rec)[1]
    for (i, syll) = enumerate(sylls)
        if any(get_mark(syll) .== trigsylls)
            trigno += 1
            measure_avail = bounds(syll)[1] - lastend
            pre_bnd = measure_avail >= pre_syll ? -pre_syll : -measure_avail
            expanded[trigno] = IntervalSet(
                RelativeInterval(
                    syll,
                    true,
                    MarkedInterval(pre_bnd, 0, get_mark(syll))
                ),
                syll
            )
            points[trigno] = NakedPoints(
                collect(bounds(syll)),
                bounds(expanded[trigno])
            )
        end
        lastend = bounds(sylls[i])[1]
    end
    resize!(expanded, trigno)
    resize!(points, trigno)
    expanded, points
end

# join sets of marked intervals, defined on super-interval dom, if intervals are
# closer than some distance max_dist, and at least one of the intervals returns
# true for seed_mark_pred, only including intervals which are true for either
# seed_mark_pred or join_mark_pred
function grow_intervals(
    dom::D,
    ints::AbstractVector{<:MarkedInterval},
    seed_mark_pred::Function,
    join_mark_pred::Function,
    max_dist::Real
) where {E, D<:NakedInterval{E}}
    nint = length(ints)
    if nint == 0
        return Vector{D}()
    end
    ints_merged = Vector{D}(undef, nint)
    full_join_pred = m -> join_mark_pred(m) | seed_mark_pred(m)

    intno = 0
    elig_idx = 1
    joined_start = bounds(ints[1])[1]
    last_end = bounds(ints[1])[2]
    growing = seed_mark_pred(get_mark(ints[1]))
    for i in 2:nint
        this_mark = get_mark(ints[i])
        b, e = bounds(ints[i])
        this_int = b - last_end

        if growing
            growing = (this_int <= max_dist) & full_join_pred(this_mark)
            if ! growing # end of growing section
                elig_index = i + 1
                intno += 1
                ints_merged[intno] = NakedInterval(joined_start, last_end)
            end

        elseif seed_mark_pred(this_mark)
            # start interval
            growing = true

            # back track: Keep moving backwards until in invalid mark / gap
            # is found
            back_i = i - 1
            last_b = b
            while (
                back_i >= elig_idx &&
                full_join_pred(get_mark(ints[back_i])) &&
                last_b - bounds(ints[back_i])[2] <= max_dist
            )
                last_b = bounds(ints[back_i])[1]
                back_i -= 1
            end

            joined_start = last_b
        end

        last_end = e
    end

    resize!(ints_merged, intno)
    ints_merged
end
