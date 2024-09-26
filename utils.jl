# random Alex utils

using LinearAlgebra    # needed for I
using FFTW

function di(x)
    """Print condensed form of a variable (REPL style) to screen"""
    io=IOBuffer();
    show(IOContext(io, :limit => true, :displaysize => (24,80)), "text/plain",x);
    println(String(take!(io)))  # yuk
end


"""
interpmat1d(t,s) interpolation matrix from source nodes s to target nodes t

Only stable for small node numbers (<100).
"""
function interpmat1d(t,s)



end


"""
nodes, weights = clencurt(n::Integer)

(n+1)-point Clenshaw-Curtis quadrature on [-1,1]. nodes in ascending order
Adapted from Barnett mpspack matlab.
"""
function clencurt(n)
    th = [i*π/n for i in 0:n]      # Chebyshev extreme nodes
    x = -cos.(th)
    W = kron(-1.0./((1:n÷2).^2 .- 0.25), [0,1])   # for either even or odd
    if n%2==1 W=[W;0.0]; end            # include extra pi-freq term if odd
    w = ifft([4.0; W; W[end-1:-1:1]])    # 4 is the zero-freq term
    w = [w[1]/2; w[2:n]; w[1]/2]       # endpoints get 1/2 weight since want 1/2 circle
    x,real.(w)
end


"""
    D,x = chebydiffmat(N) returns dense (N+1)*(N+1) differentiation matrix 
    `D` and N+1 Chebychev nodes `x`, for the standard 1D interval [-1,1].
    The matrix multiplies a vector of function values
    at these nodes to give an approximation to the vector of derivative values.
    Note: nodes are in ascending order starting at -1.0 and ending at 1.0.
    Adapted from L N Trefethen's cheb.m from "Spectral Methods in MATLAB" book.
"""
function chebydiffmat(N)
    if N==0
        x=1; D=0
    else
        x = -cos.(π*(0:N)/N)          # LNT had descending (+ sign)
        c = [2; ones(N-1); 2] .* (-1).^(0:N)
        X = x*ones(N+1)'              # duplicates nodes in each col
        dX = X-X'                     # matrix of pairwise node differences
        D = (c*(1.0./c)') ./ (dX + I)
        D = D - diagm(vec(sum(D,dims=2)))
    end
    return D,x
end


# following pulled from BIEbook ----------------
"""
    perispec_deriv!(F::Vector{T}, f::Vector{T}) where T <: Complex

Non-allocating version of `perispec_deriv`, for complex input to
complex output. Writes into 1st array.
"""
function perispec_deriv!(F::Vector{T}, f::Vector{T}) where T<:Complex
    # note we don't bother with 2x speed version for real-to-real
    N = length(f)
    kmax = (N-1)÷2
    filt = [1im*(-kmax:-1); 0; 1im*(1:kmax)]   # freqs -kmax:kmax
    if N%2==0 filt = [0; filt] end   # 0 in Nyquist keeps real for real input
    F .= ifft(fft(f) .* ifftshift(filt))     # filter f
end
"""
    F::Vector{T} = perispec_deriv(f::Vector{T})

Uses FFT to compute the vector of samples `f'` of the derivative of a given
smooth 2pi-periodic function approximated by its given vector of samples `f`.
The input and output sample points are the same, and assumed equispaced on
some 2pi-periodic interval.

The reality of output is same as input (ie, real->real, or complex->complex)
"""
function perispec_deriv(f::Vector{T}) where T<:Complex
    F = similar(f)
    perispec_deriv!(F,f)
    F
end
function perispec_deriv(f::Vector{T}) where T<:Real
    F = similar(f, complex(T))            # promote and work in complex
    perispec_deriv!(F,complex(f))
    real.(F)
end
# -------------------------------------------------


"""
x,w,nx,kappa = unitcircle(N::Integer)
    
Periodic trap rule quadrature for unit circle.
Returns x = 2*N node coords, w = N weights. 
nx is unit normals 2*N, kappa is curvatures.
The first node is at theta=0.
    
"""
function unitcircle(N::Integer)
    th = (0:N-1)/N*2*pi
    x = [cos.(th)'; sin.(th)']
    nx = copy(x)
    w = (2*pi/N)*ones(N)
    kap = -1.0*ones(N)
    x,w,nx,kap
end

"""
x,w,nx,kappa = starfish(N::Integer=100,freq::Integer=5,ampl=0.3,rot=1.0)
    
Periodic trap rule quadrature for smooth starfish.
Returns x = 2*N node coords, w = N weights. 
nx is unit normals 2*N, kappa is curvatures.
The first node is at theta=0.

*** to doc, output struct instead, etc.
"""
function starfish(N::Integer=100,freq::Integer=5,ampl=0.3,rot=1.0)
    th = (0:N-1)'/N*2*pi                      # col vec
    r = 1.0 .+ ampl*cos.(freq*th .- rot)   # set of radii. Note .operators
    rp = -ampl*freq*sin.(freq*th .- rot)
    c = cos.(th); s = sin.(th)
    x = [r.*c; r.*s]
    xp = [rp.*c.-r.*s; rp.*s.+r.*c]  # velocity
    #di(xp[:,1])
    sp = sqrt.(vec(sum(xp.^2,dims=1)))        # speed at nodes
    w = (2*pi/N)*sp                   # PTR
    nx = [xp[2,:]./sp -xp[1,:]./sp]'            # 2*N
    xpp = [perispec_deriv(xp[1,:]) perispec_deriv(xp[2,:])]'  # 2*N
    kap = vec(sum(xpp.*nx,dims=1) ./ sp'.^2)   # curvature
    x,w,nx,kap
end
