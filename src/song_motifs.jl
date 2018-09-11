struct MarkedInterval{D<:Number, M}
    interval::NTuple{2, D}
    mark::M
end

function raster_syll_data(
    sylls::AbstractVector{<:MarkedInterval},
    motif_i::AbstractArray{<:Integer},
    motif_len::Integer,
    pre = 0.05, post = 0.55
)
    n_motif = length(motif_i)
    motif_sylls, other_sylls, int_mask = destruct(
        map(
            i -> select_motif_sylls(sylls, i, i + motif_len - 1, pre, post),
            motif_i
        )
    )
end

function motif_events(event_set, int_masks)
    map(
        int_motif -> mask_events(event_set, int_motif[1], int_motif[2]),
        int_masks
    )
end

function select_motif_sylls(
    sylls::AbstractVector{<:MarkedInterval},
    motif_b::Integer,
    motif_e::Integer,
    pre::Real,
    post::Real,
    nsyll::Integer = length(sylls)
)
    motif_sylls = view(sylls, motif_b:motif_e)
    onset = motif_sylls[1].interval[1]
    int_mask = (onset - pre, onset + post)
    # The following is a hack to get around the behavior of searchsortedfirst
    # Basically the 'by' argument applies to the collection as well as the
    # element being compared.
    # The goal is to compare  the OFFSET of syllables to the ONSET of the
    # the motif range, so the 'search_interval' term below has its interval
    # swapped to get the right answer for the searches
    search_interval = (interval = (int_mask[2], int_mask[1]),)
    prior_syll_idx = searchsortedfirst(
        view(sylls, 1:(motif_b - 1)),
        search_interval,
        by = x -> x.interval[2]
    )
    n_ps = searchsortedlast(
        view(sylls, (motif_e + 1):nsyll),
        search_interval,
        by = x -> x.interval[1]
    )
    if prior_syll_idx < motif_b
        n_pr = n_ndx(prior_syll_idx, motif_b - 1)
    else
        n_pr = 0
    end
    other_sylls = Vector{MarkedInterval}(undef, n_pr + n_ps)
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
