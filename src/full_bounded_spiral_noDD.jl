# simple solve of modified Helmholtz PDE with Dirichlet BC on an arc, using
# SLP on the arc. Barnett 7/25/24.
# Now with zero-flux outer circle boundary. 
# Well-conditioned version sing Dan's idea where U'' not U is an unknown.
# Barnett 9/16/24. 
using YukBIE2D
using LinearAlgebra
using Printf
using Gnuplot
using ColorSchemes  # for gnuplot heatmaps

include("computeHelpers.jl")

verb = 1
ka = 1.3 # 0.001 #10      # aka phi, inverse decay length

# arc param on [-1,1]
# a = 0.5; b=0.1          # angular half-width, angular offset
# seg(t) = [-1.0+cos(a*t+b),sin(a*t+b)]   # fit inside unit circle
# segp(t) = [-a*sin(a*t+b),a*cos(a*t+b)]

function arcquad(seg,segp,N)   # discretize arc param by seg and deriv segp
    t,w = clencurt(N-1)       # N nodes
    sx = reduce(hcat, seg.(t))  # 2*N
    sxp = reduce(hcat, segp.(t))  # 2*N
    speed = vec(sqrt.(sum(sxp.^2,dims=1)))
    sw = w.*speed
    sx,sw,t               # note sw = speed-weights
end

function antiderivmat(Darc)  # get N*N antideriv matrix from arclen deriv mat
    U,S,V = svd(Darc)
    # take pseudoinv by killing off the last singular vector:
    Q = V[:,1:end-1] * diagm(1.0./S[1:end-1]) * U[:,1:end-1]'
    Q - ones(size(Q,1),1)*Q[1,:]'          # shift all cols to start at 0
end

function getSpiral(spacing, winding, N)

    t,w = clencurt(N-1)       # N nodes

    scrap, thetas = straightSeg(t, [0, 0], [0, winding*pi])
    #println(thetas)   # ugh, another map

    seg(t) = [spacing*t*cos(t), spacing*t*sin(t)]
    segp(t) = [spacing*cos(t) - spacing*t*sin(t), spacing*sin(t) + spacing*t*cos(t)]
    # *** Forgot you need to use chain rule: deriv w.r.t. theta not same as t!

    sx = reduce(hcat, seg.(thetas))  # 2*N
    sxp = reduce(hcat, segp.(thetas)) * (winding*pi/2) # *** since chain rule!!
    D,_ = chebydiffmat(N-1)   # N nodes
    sxpa = sx*D'          # deriv of rows
    # *** if you'd run this test would have seen sxp not compat w/ sx :( ...
    @printf "getSpiral: deriv max err on coords %.3g\n" norm(vec(sxp-sxpa),Inf)
    speed = vec(sqrt.(sum(sxp.^2,dims=1)))
    sw = w.*speed
    sx,sw,t

end


function testsegp(seg,segp,N)      # check funcs segp and seg
    t,w = clencurt(N-1)       # N nodes
    sx = reduce(hcat, seg.(t))  # 2*N
    sxp = reduce(hcat, segp.(t))  # 2*N
    D,_ = chebydiffmat(N-1)   # N nodes
    sxpa = sx*D'          # deriv of rows
    @printf "testsegp: deriv max err on coords %.3g\n" norm(vec(sxp-sxpa),Inf)
end

xtest = [-0.4,0.3]   # test pt
ng = 300 # plot grid size (N=200 ng=30 takes 0.04s to eval :)
span = 1.1
g = range(-span, span, ng)       # grid in each dim
o = ones(size(g))           # also row-vec
tx = [kron(o,g)';kron(g,o)']  # fill grid of targs (ok to fill, sim size to u)

save_dir = "saved_us/"

# number of nodes on curve
N = 120

#params = [(0.05, 5.01)]
params = [(0.1, 2)]
params = [(0.01, 11.31), (0.015, 9.22), (0.02, 7.97), (0.025, 7.12), (0.03, 6.49), (0.035, 6.0),
            (0.04, 5.61), (0.045, 5.29), (0.05, 5.01), (0.055, 4.77), (0.06, 4.57), (0.065, 4.39),
            (0.07, 4.22)]

xi = 0.2 # 0.01       # channel width scaled by diffusion ratio
alpha = 0.1    # inverse permeability

for pair in params

    if false         # select shape
        ### arc
        savename = "arc_full"
        a = 0.4; b=0.1          # angular half-width, angular offset
        seg(t) = [-1.0+cos(a*t+b),sin(a*t+b)]   # fit inside unit circle
        segp(t) = [-a*sin(a*t+b),a*cos(a*t+b)]
        sx,sw,t = arcquad(seg,segp,N)
        if verb>0 testsegp(seg,segp,N); end
    else     ### spiral
        savename = "spiral_"*string(pair[1])
        spacing, winding = pair
        sx,sw,t = getSpiral(spacing, winding, N)
    end
    # sw = arc-len weights (include speed), is a col vec
    D, nodes = chebydiffmat(N-1)       # N*N and the nodes on [-1,1]
    speed = sqrt.(sum((D*sx').^2, dims=2));   # unweighted speeds |x'| on nodes
    #println(speed); println(sw); println("sum sw = ",sum(sw))
    Iss = diagm(vec((speed).^-1))
    Darc = Iss*D;             # mat for deriv w.r.t. arclength
    Q = antiderivmat(Darc)    # mat for anti-deriv w.r.t. arclength (0 at left end)
    arclen = Q*ones(N)           # arc-length vector
    L = sum(sw)             # length of this arc
    println("L = ",L)

    # new length=N+2 row vectors to extract BCs from [sigma,a,b] vector...
    valleft = [zeros(N);1;0]'      # row takes val (left end)
    derleft = [zeros(N);0;1]'           # Here [a,b] are unknown consts for arc
    valright = [xi*(sw'*Q)';1;L]'      # note sw*Q is row vector taking Q_L.sig
    derright = [xi*sw;0;1]'            # from rep U'(s) = a + xi.(Q.sig)(s)

    Nb = 2*N               # since bdry longer than seg
    bx,bw,bnx,bcurv = unitcircle(Nb)     # discretize bdry

    S11 = YukSLPmat_selfcrude(sx,sw,ka)     # arc to self
    S12,_ = YukSLPmats(sx,[],bx,bw,ka,grad=false)    # arc from bdry
    _,DT21 = YukSLPmats(bx,bnx,sx,sw,ka)                # boundary from arc
    DT22 = YukSLPdermat_selfcrude(bx,bnx,bcurv,bw,ka)   # boundary to self

    println("size S11:", size(S11), " size Q:", size(Q))
    println("size S12:", size(S12), " size DT21:", size(DT21), " size DT22:", size(DT22))

    println("val left:", size(valleft), " size derright:", size(derright))

    ##### A is square matrix to be inverted (NOTE Alex debugged Nb vs N here:)
    # unknown (col) ordering: sigma_1, [a,b], dens_2
    # equation (block row) order: permeability on arc; BCs on arc; no-flux on gamma_2
    A = [(S11 - xi*Q^2 + alpha*diagm(ones(N)))   -ones(N) -arclen    S12;
        valleft                                                      zeros(Nb)';
        derright                                                     zeros(Nb)';
        DT21                                      zeros(Nb, 2)  0.5*diagm(ones(Nb))+DT22]

    rhs = [zeros(N); 1; 0; zeros(Nb)];  # U(0)=1, U'(L)=0: stack as col (ordering above)
    
    println("size A: ", size(A), " size rhs: ", size(rhs))
    println("condition number kappa(A): ", cond(A))

    co = A\rhs;                # solve

    dens1, ab, dens2 = co[1:N], co[N+1:N+2], co[N+3:end];     # extract answers

    U = ab[1] .+ ab[2]*arclen .+ xi*Q*(Q*dens1)              # eval rep for U(s) on nodes

    println("size dens1:", size(dens1), " size U:", size(U), " size dens2:", size(dens2))
    #println("U:", U)
    println("U at right end: ", U[end])

    u,_ = YukSLPeval(tx,[],sx,sw,dens1,ka,grad=false) .+
            YukSLPeval(tx,[],bx,bw,dens2,ka,grad=false)

    insidecirc(x,y) = x^2+y^2<1.0 ? 1.0 : 0.0
    u .*= insidecirc.(tx[1,:],tx[2,:])     # kill off vals outside circ
    # (flux cons check cannot work without this!!)

    stepx = tx[1, 2] - tx[1, 1]
    stepy = tx[2, ng+1] - tx[2, 1]
    #println("steps:", stepx, " ",  stepy)

    # use old-style BC-exctracting vectors (not really needed)
    Uvalleft = [1;zeros(N)]';      # row takes val @ left end
    Uderleft = Darc[1,:]';
    Uderright = Darc[end,:]';
    singleSegChecks(Uderleft*U, Uderright*U, u, stepx*stepy, ka, xi)

    u = reshape(u,(ng,ng))

#     save_info(save_dir*"spiral_"*string(spacing)*".h5",
#         u, 0, sx[1,:], sx[2,:], ng, span) # add phi, xi, alpha
    save_info(save_dir*savename*".h5", u, 0, sx[1,:], sx[2,:], ng, span)


end

