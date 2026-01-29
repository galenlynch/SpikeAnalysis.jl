"""
    WarpPlan{R, S<:Union{Nothing, Float64}}

Type for storing a piece of a piece-wise linear transform (warping) of points.
"""
struct WarpPlan{R,S<:Union{Nothing,Float64}}
    bounds::NTuple{2,R}
    ref::R
    offset::R
    scaling::S
end

function WarpPlan(
    start_anch::Number,
    stop_anch::Number,
    start_ref::Number,
    stop_ref::Number,
)
    WarpPlan(
        (start_anch, stop_anch),
        start_anch,
        start_ref,
        (stop_ref - start_ref) / (stop_anch - start_anch),
    )
end

function offset_warpplan(start, stop, anchor, ref_anchor)
    WarpPlan((start, stop), anchor, ref_anchor, nothing)
end

const DefWarpPlan{R} = WarpPlan{R,<:Union{Nothing,Float64}}

function piecewise_warp_plan(
    anchors::Union{AbstractVector{R},Tuple},
    ref_anchors::Union{AbstractVector{R},Tuple},
    start::R,
    stop::R,
) where {R}
    na = length(anchors)
    na > 0 || throw(ArgumentError("Anchors is empty"))
    if length(ref_anchors) != na
        throw(ArgumentError("anchors and ref_anchors not the same size"))
    end
    plan = Vector{DefWarpPlan{R}}(undef, na + 1)
    plan[1] = offset_warpplan(start, anchors[1], anchors[1], ref_anchors[1])
    for anchor_no = 2:na
        plan[anchor_no] = WarpPlan(
            anchors[anchor_no-1],
            anchors[anchor_no],
            ref_anchors[anchor_no-1],
            ref_anchors[anchor_no],
        )
    end
    plan[end] = offset_warpplan(anchors[end], stop, anchors[end], ref_anchors[end])
    plan
end

function apply_warpplan!(pts, plan::AbstractVector{<:WarpPlan})
    nwarp = length(plan)
    nwarp > 0 || throw(ArgumentError("plan cannot be empty"))
    npt = length(pts)
    i_b = searchsortedfirst(pts, plan[1].bounds[1])
    last_i = i_b - 1
    for plan_no = 1:nwarp
        last_i == npt && break
        i_e = searchsortedlast(pts, plan[plan_no].bounds[2])
        rng = (last_i+1):i_e
        pts[rng] .= linear_pt_warp.(pts[rng], Ref(plan[plan_no]))
        last_i = i_e
    end
    pts
end

# Assumes sorted
function piecewise_pt_warp!(
    pts::AbstractVector{<:Number},
    anchors,
    ref_anchors,
    start = nothing,
    stop = nothing,
)
    isempty(anchors) && throw(ArugmentError("anchors cannot be empty"))
    pts_empty = isempty(pts)
    if start == nothing
        start = pts_empty ? anchors[1] : min(pts[1], anchors[1])
    end
    if stop == nothing
        stop = pts_empty ? anchors[end] : max(pts[end], anchors[end])
    end
    plan = piecewise_warp_plan(anchors, ref_anchors, start, stop)
    apply_warpplan!(pts, plan)
end

function piecewise_pt_warp(pts::AbstractVector{<:Number}, anchors, ref_anchors, args...)
    piecewise_pt_warp!(copy(pts), anchors, ref_anchors, args...)
end

function piecewise_pt_warp(
    spkset::AbstractVector{A},
    trigset::TrigSet,
    args...;
    ex_no::Integer = div(length(trigset.trigs), 2),
) where {A<:AbstractVector{<:Number}}
    n_set = length(spkset)
    ref_intervals = bounds.(getfield.(trigset.trigs[ex_no].triggered, :interval))
    ref_anchors = collect(Iterators.flatten(ref_intervals))
    warped = Vector{div_type(A)}(undef, n_set)
    for i = 1:n_set
        warped[i] = piecewise_pt_warp(
            spkset[i],
            collect(
                Iterators.flatten(
                    bounds.(getfield.(trigset.trigs[i].triggered, :interval)),
                ),
            ),
            ref_anchors,
            args...,
        )
    end
    warped
end

linear_pt_warp(pt, ref, offset, scale) = scale .* (pt .- ref) .+ offset
linear_pt_warp(pt, ref, offset) = offset .+ pt .- ref
function linear_pt_warp(pt, plan::WarpPlan{<:Any,Nothing})
    linear_pt_warp(pt, plan.ref, plan.offset)
end
function linear_pt_warp(pt, plan::WarpPlan{<:Any,<:Real})
    linear_pt_warp(pt, plan.ref, plan.offset, plan.scaling)
end

function func_warp_remove!(
    ifr_out,
    ifr_basis::Union{AbstractVector,AbstractRange},
    interp::AbstractInterpolation,
    anchors::Union{AbstractVector,Tuple},
    ref_anchors::Union{AbstractVector,Tuple},
    start,
    stop,
)
    plan = piecewise_warp_plan(anchors, ref_anchors, start, stop)
    for warpsection in plan
        warp_ifr_section_remove!(ifr_out, interp, ifr_basis, warpsection)
    end
    ifr_out
end

function func_warp_remove!(
    ifr_out,
    ifr_basis::Union{AbstractRange,AbstractVector},
    ref_ifr::AbstractVector{<:Number},
    ref_ifr_basis::Union{AbstractVector,AbstractRange},
    args...,
)
    interp = LinearInterpolation(ref_ifr_basis, ref_ifr)
    func_warp_remove!(ifr_out, ifr_basis, interp, args...)
end

func_warp_remove(ifr_in, args...) = func_warp_remove!(copy(ifr_in), args...)

# Warp the original basis onto the template basis, find the interpolated
# template values, and then subtract the interpolated values from the original
# points. Additionally, scale the interpolated values to keep the integral
# constant.
#
# Since we are warping from the original to the template, scaling
# means multiplying by the scale factor. To clarify, if the original syllable is
# longer than the template, then the scale factor will be less than one. Once
# the interpolated values are known, we want to decrease their amplitude beacuse
# it has a larger basis in the original data. Therefore, multiply by the scaling
# factor to decrease its amplitude
function warp_ifr_section_remove!(
    ifr_out,
    interp::AbstractInterpolation,
    ifr_basis,
    warpsection,
)
    i_b = searchsortedfirst(ifr_basis, warpsection.bounds[1])
    i_e = searchsortedlast(ifr_basis, warpsection.bounds[2])
    warped_basis = linear_pt_warp(ifr_basis[i_b:i_e], warpsection)
    if warpsection.scaling == nothing
        ifr_out[i_b:i_e] .-= interp.(warped_basis)
    else
        ifr_out[i_b:i_e] .-= interp.(warped_basis) .* warpsection.scaling # Keep area the same
    end
    ifr_out
end

function func_remove_mean!(
    ifr_out::AbstractVector,
    ifr_basis::Union{AbstractVector,AbstractRange},
    mean_ifr::AbstractVector,
    mean_basis::Union{AbstractVector,AbstractRange},
    observed_intervals::AbstractVector{<:Union{AbstractVector,Tuple}},
    ref_interval::Union{AbstractVector,Tuple},
    interval_pre::Real,
    interval_post::Real = 0,
)
    n_int = length(observed_intervals)
    interp = LinearInterpolation(mean_basis, mean_ifr)
    for i = 1:n_int
        func_warp_remove!(
            ifr_out,
            ifr_basis,
            interp,
            observed_intervals[i],
            ref_interval,
            observed_intervals[i][1] - interval_pre,
            observed_intervals[i][2] + interval_post,
        )
    end
    ifr_out
end

func_remove_mean(ifr_in, args...) = func_remove_mean!(copy(ifr_in), args...)
