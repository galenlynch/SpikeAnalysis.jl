function xcorr_normed(
    u::AbstractArray{<:AbstractFloat},
    v::AbstractArray{<:AbstractFloat};
    eltype::Type = Float32,
    center::Bool = true
)
    nu = length(u)
    nv = length(v)
    nlag = max(nu, nv) - 1
    n = 2 * nlag + 1
    n_padded = nextfastfft(n)
    n_comp = div(n_padded, 2) + 1

    r_buf = Vector{eltype}(undef, n_padded)
    uim_buf = Vector{Complex{eltype}}(undef, n_comp)
    vim_buf = similar(uim_buf)
    accum = zeros(eltype, n_padded)
    out = similar(accum, n)
    p = plan_rfft(r_buf)
    ip = plan_irfft(uim_buf, n_padded)

    out_norm = _xcorr!(
        accum, u, nu, v, nv, r_buf, uim_buf, vim_buf, p, ip, center
    )
    xcorr_unwrap!(out, accum, nlag, n_padded)
    out ./= out_norm
    out
end

function _xcorr_normed!(
    out::AbstractVector{<:Real},
    accum::AbstractVector{<:Real},
    r_buf::AbstractVector{<:Real},
    uim_buf::AbstractVector{<:Complex},
    vim_buf::AbstractVector{<:Complex},
    us::AbstractArray{<:AbstractArray{<:Real}},
    vs::AbstractArray{<:AbstractArray{<:Real}},
    nu::Integer,
    u_ls::AbstractVector{<:Integer},
    v_ls::AbstractVector{<:Integer},
    p,
    ip,
    n_padded::Integer,
    nlag::Integer,
    center::Bool = true
)

    accum .= 0
    out_norm = zero(eltype(out))
    for vecno = 1:nu
        out_norm += _xcorr!(
            accum,
            us[vecno],
            u_ls[vecno],
            vs[vecno],
            v_ls[vecno],
            r_buf,
            uim_buf,
            vim_buf,
            p,
            ip,
            center
        )
    end

    xcorr_unwrap!(out, accum, nlag, n_padded)
    out ./= out_norm * n_padded
    out
end

function prepare_xcorr(
    us::AbstractArray{<:AbstractArray{<:Real}},
    vs::AbstractArray{<:AbstractArray{<:Real}},
    outtype::Type = Float32,
)
    nu = length(us)
    nu == length(vs) || throw(ArgumentError("us and vs not the same length"))
    u_ls = length.(us)
    u_max_l = maximum(u_ls)
    v_ls = length.(vs)
    v_max_l = maximum(v_ls)
    nlag = max(u_max_l, v_max_l) - 1
    n = 2 * nlag + 1

    n_padded = nextfastfft(n)
    n_comp = div(n_padded, 2) + 1

    r_buf = Vector{outtype}(undef, n_padded)
    accum = similar(r_buf)
    uim_buf = Vector{Complex{outtype}}(undef, n_comp)
    vim_buf = similar(uim_buf)
    out = Vector{outtype}(undef, n)

    p = plan_rfft(r_buf)
    ip = plan_brfft(uim_buf, n_padded)

    return (
        out,
        accum,
        r_buf,
        uim_buf,
        vim_buf,
        nu,
        u_ls,
        v_ls,
        p,
        ip,
        n_padded,
        nlag
    )
end

function xcorr_normed(
    us::AbstractArray{<:AbstractArray{<:Real}},
    vs::AbstractArray{<:AbstractArray{<:Real}};
    outtype::Type = Float32,
    center::Bool = true
)
    out, accum, r_buf, uim_buf, vim_buf, nu, u_ls, v_ls, p, ip, n_padded, nlag =
        prepare_xcorr(us, vs, outtype)
    _xcorr_normed!(
        out,
        accum,
        r_buf,
        uim_buf,
        vim_buf,
        us,
        vs,
        nu,
        u_ls,
        v_ls,
        p,
        ip,
        n_padded,
        nlag,
        center
    )
end

# Mutates r_buf and im_buf
# before call, r_buf should have padded data
# After call, r_buf will have centered data
# im_buf will be overwritten to have the fourier transform of r_buf after call
function pad_transform!(u, ulen, r_buf, im_buf, p, center)
    copyto!(r_buf, 1, u, 1, ulen)
    r_buf[(ulen + 1):end] .= 0
    if center
        r_buf[1:ulen] .-= mean(u)
    end
    mul!(im_buf, p, r_buf)
end

# Kernel for xcorr
function _xcorr!(
    accum, u, ulen, v, vlen, r_buf, uim_buf, vim_buf, p, ip, center
)
    # Transform u
    pad_transform!(u, ulen, r_buf, uim_buf, p, center)
    unorm = sum(abs2, view(r_buf, 1:ulen))
    # Transform v
    pad_transform!(v, vlen, r_buf, vim_buf, p, center)
    vnorm = sum(abs2, view(r_buf, 1:vlen))
    # Multiply to do cross correlation in fourier domain
    uim_buf .= uim_buf .* conj.(vim_buf)
    # inverse fourier transform
    mul!(r_buf, ip, uim_buf)
    # Add to accumulator
    accum .+= r_buf
    # return norm for this xcorr
    sqrt(unorm * vnorm)
end

function xcorr_unwrap!(out, accum, nlag, n_padded)
    copyto!(out, nlag + 1, accum, 1, nlag + 1)
    copyto!(out, 1, accum, ndx_offset(n_padded, -nlag), nlag)
    out
end

function xcorr_basis(nu::Integer, nv::Integer)
    maxlag = max(nu, nv) - 1
    -maxlag:maxlag
end

function xcorr_basis(nxcorr)
    maxlag = div(nxcorr, 2)
    -maxlag:maxlag
end

function xcorr_sig(
    ib::Integer,
    ie::Integer,
    xc::AbstractVector{<:Number},
    xc_nulls::AbstractVector{<:Number},
    ntrial = length(null_xcs)
)
    xc_extr_idx = argmax(abs.(view(xc, ib:ie)))
    xc_extr = xc[ndx_offset(ib, xc_extr_idx)]
    mc_twotail_asymm_p(xc_extr, xc_nulls, ntrial)
end

function xcorr_sig(
    chunks_u::AbstractVector{<:SharedVector},
    chunks_v::AbstractVector{<:SharedVector},
    xcs_null::AbstractVector{<:Number},
    ib::Integer,
    ie::Integer,
    n_trial = length(xcs_null),
    center::Bool = false
)
    xc = xcorr_normed(chunks_u, chunks_v, center = center)
    xcorr_sig(ib, ie, xc, xcs_null, n_trial)
end

function xcorr_sig(
    chunks_u::AbstractVector{<:SharedVector},
    chunks_v::AbstractVector{<:SharedVector},
    ib::Integer,
    ie::Integer,
    n_trial = 1000,
    center::Bool = false
)
    xcs_null = xcorr_null_mc(chunks_u, chunks_v, ib, ie, n_trial, center)
    xcorr_sig(chunks_u, chunks_v, xcs_null, ib, ie, n_trial, center)
end

function xcorr_null_mc(
    chunks_u::AbstractVector{<:SharedVector{E}},
    chunks_v::AbstractVector{<:SharedVector{F}},
    ib,
    ie,
    n_trial = 1000,
    center::Bool = false
) where {E, F}
    outtype = promote_type(E, F)
    nu = length(chunks_u)
    length(chunks_v) == nu || throw(ArgumentError("Chunks not the same length"))
    extr_out = Vector{outtype}(undef, n_trial)
    (
        xc_out,
        accum,
        r_buf,
        uim_buf,
        vim_buf,
        nu,
        u_ls,
        v_ls,
        pl,
        ipl,
        n_padded,
        nlag
    ) = prepare_xcorr(chunks_u, chunks_v)

    for i = 1:n_trial
        p = randperm_notsame(nu)
        shuff_vs = chunks_v[p]
        shuff_v_ls = v_ls[p]
        _xcorr_normed!(
            xc_out,
            accum,
            r_buf,
            uim_buf,
            vim_buf,
            chunks_u,
            shuff_vs,
            nu,
            u_ls,
            shuff_v_ls,
            pl,
            ipl,
            n_padded,
            nlag,
            center
        )
        xc_max = typemin(eltype(xc_out))
        xc_min = typemax(eltype(xc_out))
        for j = ib:ie
            xc_max = max(xc_max, xc_out[j])
            xc_min = min(xc_min, xc_out[j])
        end
        extr_out[i] = ifelse(abs(xc_max) > abs(xc_min), xc_max, xc_min)
    end
    extr_out
end
