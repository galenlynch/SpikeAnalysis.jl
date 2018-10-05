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

# Different output basis!
function xcorr_normed(
    us::AbstractArray{<:AbstractArray{<:Real}},
    vs::AbstractArray{<:AbstractArray{<:Real}};
    eltype::Type = Float32,
    center::Bool = true
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

    r_buf = Vector{eltype}(undef, n_padded)
    uim_buf = Vector{Complex{eltype}}(undef, n_comp)
    vim_buf = similar(uim_buf)
    accum = zeros(eltype, n_padded)
    out = Vector{eltype}(undef, n)

    p = plan_rfft(r_buf)
    ip = plan_irfft(uim_buf, n_padded)

    out_norm = zero(eltype)

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
    out ./= out_norm
    out
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
