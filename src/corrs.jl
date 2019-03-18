# Version for single vectors
function xcorr_normed_valid(
    u::AbstractArray{<:AbstractFloat},
    v::AbstractArray{<:AbstractFloat},
    nlag::Integer;
    outeltype::Type = Float32,
    center::Bool = true
)
    ulen = length(u)
    length(v) == ulen || error("Don't know what to do")
    ulen > 2 * nlag || error("u is not long enough")
    (
        out,
        accum,
        r_buf,
        uim_buf,
        vim_buf,
        p,
        ip,
        bufflen
    ) = prepare_xcorr(ulen, nlag, outeltype)
    out_norm = _xcorr!(
        accum, u, ulen, v, r_buf, bufflen, uim_buf, vim_buf, p, ip, nlag, center
    )
    xcorr_unwrap!(out, accum, nlag, bufflen)
    out ./= out_norm
    out
end

# Version for many vectors
function xcorr_normed_valid(
    us::AbstractArray{<:AbstractArray{<:Real}},
    vs::AbstractArray{<:AbstractArray{<:Real}},
    nlag::Integer;
    outtype::Type = Float32,
    center::Bool = true
)
    out, accum, r_buf, uim_buf, vim_buf, p, ip, bufflen, nu, u_ls =
        prepare_xcorr(us, vs, nlag, outtype)
    _xcorr_normed_valid!(
        out,
        accum,
        r_buf,
        uim_buf,
        vim_buf,
        us,
        vs,
        nu,
        u_ls,
        p,
        ip,
        bufflen,
        nlag,
        center
    )
end

function _xcorr_normed_valid!(
    out::AbstractVector{<:Real},
    accum::AbstractVector{<:Real},
    r_buf::AbstractVector{<:Real},
    uim_buf::AbstractVector{<:Complex},
    vim_buf::AbstractVector{<:Complex},
    us::AbstractArray{<:AbstractArray{<:Real}},
    vs::AbstractArray{<:AbstractArray{<:Real}},
    nu::Integer,
    u_ls::AbstractVector{<:Integer},
    p,
    ip,
    bufflen::Integer,
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
            r_buf,
            bufflen,
            uim_buf,
            vim_buf,
            p,
            ip,
            nlag,
            center
        )
    end

    xcorr_unwrap!(out, accum, nlag, bufflen)
    out ./= out_norm
    out
end

function prepare_xcorr(nu, nlag, outeltype)
    bufflen = nextfastfft(2 * nu - 1)
    complen = div(bufflen, 2) + 1
    outlen = 2 * nlag + 1

    r_buf = Vector{outeltype}(undef, bufflen)
    out = similar(r_buf, outlen)
    accum = zeros(outeltype, bufflen)

    uim_buf = Vector{Complex{outeltype}}(undef, complen)
    vim_buf = similar(uim_buf)

    p = plan_rfft(r_buf)
    ip = plan_irfft(uim_buf, bufflen)

    return (
        out,
        accum,
        r_buf,
        uim_buf,
        vim_buf,
        p,
        ip,
        bufflen
    )
end

function prepare_xcorr(
    us::AbstractArray{<:AbstractArray{<:Real}},
    vs::AbstractArray{<:AbstractArray{<:Real}},
    nlag::Integer,
    outtype::Type = Float32,
)
    nu = length(us)
    nu == length(vs) || throw(ArgumentError("us and vs not the same length"))
    u_ls = length.(us)
    all(u_ls .== length.(vs)) || throw(ArgumentError("elements of us must be same len as vs"))
    u_max_l = maximum(u_ls)

    (
        out,
        accum,
        r_buf,
        uim_buf,
        vim_buf,
        p,
        ip,
        bufflen
    ) = prepare_xcorr(u_max_l, nlag, outtype)

    return (
        out,
        accum,
        r_buf,
        uim_buf,
        vim_buf,
        p,
        ip,
        bufflen,
        nu,
        u_ls
    )
end

# Mutates r_buf and im_buf
# before call, r_buf should have padded data
# After call, r_buf will have centered data
# im_buf will be overwritten to have the fourier transform of r_buf after call
function pad_transform!(u, ulen, r_buf, buflen, im_buf, p, nlag, center)
    adj_len = ulen - 2 * nlag
    center_ndx = div(ulen - 1, 2) + 1
    nleft = div(adj_len - 1, 2)
    nright = adj_len - nleft
    copyto!(r_buf, 1, u, center_ndx, nright)
    left_ndx = buflen - nleft + 1
    copyto!(r_buf, left_ndx, u, 1 + nlag, nleft)
    r_buf[nright + 1:left_ndx - 1] .= 0
    if center
        mu = mean(u)
        r_buf[1:nright] .-= mu
        r_buf[left_ndx:buflen] .-= mu
    end
    mul!(im_buf, p, r_buf)
end

function rbuf_norm(r_buf, nminright, minleftndx, buflen)
    sum(abs2, view(r_buf, 1:nminright)) +
        sum(abs2, view(r_buf, minleftndx:buflen))
end

# Kernel for xcorr
function _xcorr!(
    accum, u, ulen, v, r_buf, buflen, uim_buf, vim_buf, p, ip, nlag, center
)
    adj_len = ulen - 2 * nlag
    min_side_l = div(adj_len - 1, 2)
    nminright = adj_len - min_side_l
    minleftndx = buflen - min_side_l + 1
    # Transform u
    pad_transform!(u, ulen, r_buf, buflen, uim_buf, p, nlag, center)
    unorm = rbuf_norm(r_buf, nminright, minleftndx, buflen)
    # Transform v
    pad_transform!(v, ulen, r_buf, buflen, vim_buf, p, 0, center)
    vnorm = rbuf_norm(r_buf, nminright, minleftndx, buflen)
    # Multiply to do cross correlation in fourier domain
    uim_buf .*= conj.(vim_buf)
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

function xcorr_valid_sig(
    xc::AbstractVector{<:Number},
    xc_nulls::AbstractVector{<:Number},
    ntrial = length(null_xcs)
)
    xcmin, xcmax = extrema(xc)
    xc_extr = ifelse(abs(xcmin) > abs(xcmax), xcmin, xcmax)
    mc_twotail_asymm_p(xc_extr, xc_nulls, ntrial)
end

function xcorr_valid_sig(
    nlag::Integer,
    chunks_u::AbstractVector{<:AbstractVector},
    chunks_v::AbstractVector{<:AbstractVector},
    xcs_null::AbstractVector{<:Number},
    n_trial = length(xcs_null),
    center::Bool = false
)
    xc = xcorr_normed_valid(chunks_u, chunks_v, nlag, center = center)
    xcorr_valid_sig(xc, xcs_null, n_trial)
end

function xcorr_valid_sig(
    nlag::Integer,
    chunks_u::AbstractVector{<:AbstractVector},
    chunks_v::AbstractVector{<:AbstractVector},
    n_trial = 1000,
    center::Bool = false
)
    xcs_null = xcorr_null_mc(chunks_u, chunks_v, nlag, n_trial, center)
    xcorr_valid_sig(nlag, chunks_u, chunks_v, xcs_null, n_trial, center)
end

function xcorr_null_mc(
    us::AbstractVector{<:AbstractVector},
    vs::AbstractVector{<:AbstractVector},
    nlag::Integer,
    n_trial::Integer = 1000;
    outtype::Type = Float32,
    center::Bool = false
)
    extr_out = Vector{outtype}(undef, n_trial)
    out, accum, r_buf, uim_buf, vim_buf, p, ip, bufflen, nu, us_ls =
        prepare_xcorr(us, vs, nlag, outtype)

    for i = 1:n_trial
        rp = randperm_notsame(nu)
        shuff_vs = vs[rp]
        _xcorr_normed_valid!(
            out,
            accum,
            r_buf,
            uim_buf,
            vim_buf,
            us,
            shuff_vs,
            nu,
            us_ls,
            p,
            ip,
            bufflen,
            nlag,
            center
        )
        xc_min, xc_max = extrema(out)
        extr_out[i] = ifelse(abs(xc_max) > abs(xc_min), xc_max, xc_min)
    end
    extr_out
end
