"""
    RasterTrigSylls{D, M<:MarkedInterval{D, <:Any}}

Represents a "trigger", which is a sequence of syllables in a longer sequence
that matches some pattern. This is akin to the trigger of an oscilloscope, or
electro_gui.
"""
struct RasterTrigSylls{D, M<:MarkedInterval{D, <:AbstractString}}
    triggered::Vector{M}
    other::Vector{M}
    motif_interval::NTuple{2, D}
    expanded_interval::NTuple{2, D}
end

"""
    TrigSet{D, R<:RasterTrigSylls{D, <:Any}

Represents a group of triggers, with some time around each trigger.

See Also: [`RasterTrigSylls`](@ref)
"""
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
    durs_syl
)
    # Get motif info
    motif_len = length(motif)
    motif_i = find_subseq(motif, labels_syl)
    motif_ons, motif_offs, motif_durs = motif_info(
        offs_syl, durs_syl, motif_len, motif_i
    )
    p = sortperm(motif_durs, rev = true)
    return motif_i[p], motif_ons[p], motif_offs[p], motif_durs[p]
end

function find_motifs(
    motif::AbstractVector{<:AbstractString},
    syls::AbstractVector{<:MarkedInterval}
)
    nsyl = length(syls)
    labels = get_mark.(syls)
    offs_syl = Vector{Float64}(undef, nsyl)
    durs_syl = similar(offs_syl)
    for (i, s) in enumerate(syls)
        offs_syl[i], te = bounds(s)
        durs_syl[i] = te - offs_syl[i]
    end
    find_motifs(motif, labels, offs_syl, durs_syl)
end

"""
Finds bouts of song
"""
function call_bouts(
    rec_int,
    ints_syll,
    seed_marks,
    join_marks,
    max_dist,
    contraction = max_dist
)
    bouts = grow_intervals(rec_int, ints_syll, seed_marks, join_marks, max_dist)
    silence_intervals = shrink(complement(rec_int, bouts), contraction)
    return bouts, silence_intervals
end

"""
    trig_data(
        trig_seq::AbstractVector{<:AbstractString},
        labels_syl::AbstractVector{<:AbstractString},
        offs_syl::AbstractVector{<:Real},
        durs_syl::AbstractVector{<:Real},
        syls::AbstractVector{<:MarkedInterval},
        pre::Real,
        post_expand::Union{Real, Nothing} = nothing;
        post::Union{Nothing, Real} = nothing
    )

Finds occurrences of the sub-sequence specified by `trig_seq` in the sequence
`labels_syl`. Returns a [`TrigSet`](@ref). `post` specifies time to stop after
trig onset, while `post_expand` specifies time stop after trig offset.
"""
function trig_data(
    trig_seq::AbstractVector{<:AbstractString},
    syls::AbstractVector{<:MarkedInterval},
    pre::Real,
    post_expand::Union{Real, Nothing} = nothing;
    post::Union{Nothing, Real} = nothing
)
    trig_len = length(trig_seq)
    motif_i, motif_ons, motif_offs, motif_durs = find_motifs(trig_seq, syls)

    median_motif_dur = isempty(motif_durs) ? NaN : median(motif_durs)
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

"""
    align_trigs(raw::TrigSet)

Takes a `TrigSet`, and aligns each `RasterTrigSylls` therein to the onset time
of its first triggered syllable.

See also: [`align_trigs(::RasterTrigSylls)`](@ref)
"""
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

"""
    align_trigs(raw::RasterTrigSylls)

Takes a `RasterTrigSylls`, and aligns its syllable times to the onset of the
first triggered syllable. Returns a new, aligned, `RasterTrigSylls`.

See also: [`align_trigs(::TrigSet)`](@ref)
"""
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

"""
    clip_trigs(raw::TrigSet)

Clips syllables in a TrigSet to its expanded interval.
"""
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

"""
    function add_pres(
        rec::NakedInterval{E},
        sylls::AbstractVector{M},
        trigsylls::AbstractVector{<:String};
        pre_syll::Real = 0.03,
    ) where {E, M<:MarkedInterval{E, <:Any}} -> expanded, points

Expand syllables, `syll` with label in `trigsylls`, by `pre_syll`, up to the
last syll or the beginning of the `rec` interval.

Requires input to be sorted and not overlapping.

Returns a tuple where the first elements, `expanded`, is an array of interval
sets with the original syllable joined with the "pre" period. The second element,
`points`, is an array of [`NakedPoints`](@ref) with points for syllable onset
and offset defined on an expanded interval, including the "pre" period.
"""
function add_pres(
    rec::NakedInterval{E},
    sylls::AbstractVector{M},
    trigsylls::AbstractVector{<:String};
    pre_syll::Real = 0.03,
) where {E, M<:MarkedInterval{E, <:Any}}
    intervals_are_ordered(bounds.(sylls)) || error("sylls are not well-ordered")
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
        lastend = bounds(sylls[i])[2]
    end
    resize!(expanded, trigno)
    resize!(points, trigno)
    expanded, points
end

# join sets of marked intervals, defined on super-interval dom, if intervals are
# closer than some distance max_dist, and at least one of the intervals returns
# true for seed_marks, only including intervals which are true for either
# seed_marks or join_marks
function grow_intervals(
    dom::D,
    ints::AbstractVector{<:MarkedInterval},
    seed_marks::AbstractVector,
    join_marks::AbstractVector,
    max_dist::Real
) where {E, D<:NakedInterval{E}}
    nint = length(ints)
    if nint == 0
        return Vector{D}()
    end
    ints_merged = Vector{D}(undef, nint)
    all_marks = vcat(seed_marks, join_marks)

    intno = 0
    elig_idx = 1
    joined_start = bounds(ints[1])[1]
    last_end = bounds(ints[1])[2]
    growing = anyeq(get_mark(ints[1]), seed_marks)
    for i in 2:nint
        this_mark = get_mark(ints[i])
        b, e = bounds(ints[i])
        this_int = b - last_end

        if growing
            growing = (this_int <= max_dist) & anyeq(this_mark, all_marks)
            if ! growing # end of growing section
                elig_index = i + 1
                intno += 1
                ints_merged[intno] = NakedInterval(joined_start, last_end)
            end

        elseif anyeq(this_mark, seed_marks)
            # start interval
            growing = true

            # back track: Keep moving backwards until in invalid mark / gap
            # is found
            back_i = i - 1
            last_b = b
            while (
                back_i >= elig_idx &&
                anyeq(get_mark(ints[back_i]), all_marks) &&
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

sound_loudness(s) = 10 * log10.(s .^ 2 .+ eps())
sound_loudness(s, b) = sound_loudness(filtfilt(b, s))
function sound_loudness(s, fs, band_start, band_stop, order)
    b = make_bandpass(fs, band_start, band_stop, order)
    sound_loudness(s, b)
end

function song_rhythm_power(
    l::AbstractArray,
    sr::Number,
    song_window = (5, 12),
    n = sr,
    n_ov = round(Int, 0.75 * n)
)
    s = spectrogram(l .- mean(l), n, n_ov, fs = sr, window = blackman(n))
    songmask = song_window[1] .<= s.freq .<= song_window[2]
    song_pow = sum(s.power[songmask, :], dims = 1)
    maximum(song_pow)
end

function song_rhythm_power(b::AbstractArray, s::AbstractArray, args...)
    l = sound_loudness(s, b)
    song_rhythm_power(l, args...)
end

@enum SongGuess not_song maybe_song probably_song

function guess_if_score_is_song(
    disc_proj;
    disc_slope = -5.075132275132278,
    disc_offset = -234.00380952380965,
    confident_thresh = 13.6,
    disc_thresh = 0
)
    ifelse(
        disc_proj <= disc_thresh,
        not_song,
        ifelse(
            disc_proj >= confident_thresh,
            probably_song,
            maybe_song
        )
    )
end

function score_array_for_song(
    l::AbstractArray,
    sr::Number;
    song_window = (5, 12),
    n = sr,
    n_ov = round(Int, 0.75 * n),
    n_av = convert(Int, div(sr, 2))
)
    nl = length(l)
    adj_n = min(n, nl)
    adj_n_ov = min(n_ov, round(Int, 0.75 * adj_n))
    adj_n_av = min(n_av, nl)
    rhythm_pow = song_rhythm_power(l, sr, song_window, n, n_ov)
    mean_i = maximum(moving_sum(l, adj_n_av)) / adj_n_av
    mean_i, rhythm_pow
end

function score_array_for_song(
    b::AbstractArray, s::AbstractArray, args...; kwargs...
)
    score_array_for_song(sound_loudness(s, b), args...; kwargs...)
end

function distance_from_discriminant(
    mean_i,
    rhythm_pow,
    disc_slope = -5.075132275132278,
    disc_offset = -234.00380952380965
)
    - (disc_slope * mean_i - rhythm_pow + disc_offset) /
        sqrt(disc_slope ^ 2 + 1)
end

function song_discriminant(
    l::AbstractArray,
    sr::Number;
    song_window = (5, 12),
    n = sr,
    n_ov = round(Int, 0.75 * n),
    disc_slope = -5.075132275132278,
    disc_offset = -234.00380952380965
)
    intensity, rhythm = score_array_for_song(
        l, sr, song_window = song_window, n = n, n_ov = n_ov
    )
    distance_from_discriminant(intensity, rhythm, disc_slope, disc_offset)
end

function song_discriminant(
    b::AbstractArray, s::AbstractArray, args...;
    kwargs...
)
    song_discriminant(sound_loudness(s, b), args...; kwargs...)
end

function guess_if_array_contains_song(
    l::AbstractArray,
    sr::Number;
    song_window = (5, 12),
    n = sr,
    n_ov = round(Int, 0.75 * n),
    kwargs...
)
    disc_proj = song_discriminant(
        l, sr, song_window = song_window, n = n, n_ov = n_ov
    )
    guess_if_score_is_song(mean_i, rhythm_pow; kwargs...)
end

function guess_if_array_contains_song(
    b::AbstractArray, s::AbstractArray, args...; kwargs...
)
    guess_if_array_contains_song(sound_loudness(s, b), args...; kwargs...)
end
