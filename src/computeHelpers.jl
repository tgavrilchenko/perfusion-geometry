using HDF5

#####
## helper functions for running a network in computeField.jl
#####


################################################################
# segment structure
Base.@kwdef struct Segment
    N::Int64                    # number of points in the parameterization
    pos::Int64                  # position in the list of all segments
    x::Vector{AbstractFloat}
    y::Vector{AbstractFloat}
    w::Vector{AbstractFloat}
    D::Matrix{AbstractFloat}

    pts::Matrix{AbstractFloat} = [x y]';
    speed::Vector{AbstractFloat} = vec(sqrt.(sum((D*pts').^2, dims=2)));
    mod_w::Vector{AbstractFloat} = w.*speed;
    Darc::Matrix{AbstractFloat} = diagm(vec(ones(N+1)./speed))*D;

    head_pos::Int64 = (pos-1)*N + pos       # assumes that all segments are of size N
    tail_pos::Int64 = head_pos + N

end

################################################################
# tree structure: has a root, leaves, and vertices
Base.@kwdef struct Tree
    root::Int64
    leaves::Vector{Int64}
    vertices::Vector{AbstractArray}
end

################################################################
# plot the tree structure
function plotTree(segments, saveFileName)

    xs = Any[]
    ys = Any[]

    for seg in segments
        push!(xs, seg.x)
        push!(ys, seg.y)
    end

    tree_plt = plot(xs, ys,
            xlim = (-2.5, 2.5), ylim = (-2.5, 2.5), color = :black, grid=false,
            thickness_scaling = 1.75, legend = false, size = (400, 400))

    savefig(tree_plt, saveFileName*".pdf")
end

# t is 1D line starting at -1 and ending at 1
# rescale it to a 2D line segment from start to stop
function straightSeg(t, start, stop)
    x1, y1 = start
    x2, y2 = stop

    x = (x2-x1)/2*t + (x2+x1)/2*ones(size(t))

    if x1 != x2
        y = x*(y2-y1)/(x2-x1) + (x2*y1-x1*y2)/(x2-x1)*ones(size(t))
    else
        y = (y2-y1)/2*t + (y2+y1)/2*ones(size(t))
    end

    return x, y
end

function save_info(saveName, u, U, x, y, ng, span)#, phi, xi, alpha)

    all_keys = ["u(x)", "U(s)", "xs", "ys", "ng", "span"]#, "phi", "xi", "alpha"]

    var = try
        h5open(saveName, "r+") do f

            for key in all_keys
                if haskey(f, key)
                    delete_object(f, key)
                end
            end

            f["u(x)"] = u
            f["U(s)"] = U
            f["xs"] = x
            f["ys"] = y
            f["ng"] = ng
            f["span"] = span
#             f["phi"] = phi
#             f["xi"] = xi
#             f["alpha"] = alpha
            #f["input_flux"] = input_flux

        end
    catch e
        if isa(e,ErrorException)
                h5write(saveName, "u(x)", u)
                h5write(saveName, "U(s)", U)
                h5write(saveName, "xs", x)
                h5write(saveName, "ys", y)
                h5write(saveName, "ng", ng)
                h5write(saveName, "span", span)
#                 h5write(saveName, "phi", phi)
#                 h5write(saveName, "xi", xi)
#                 h5write(saveName, "alpha", alpha)
                #h5write(saveName, "input_flux", input_flux)
            else
                throw(e)
            end
    end

end

function singleSegChecks(input_flux, output_flux, u, dA, phi, xi)

    checkUoutput= true
    checkuConservation = true

    absorbed_flux = sum(u)*dA*(phi^2)*xi;
    out_in_ratio = -absorbed_flux/input_flux;
    println("flux cons error: ", 1-out_in_ratio,
        "    flux at right end: ", output_flux)

    if abs(out_in_ratio-1) > 1e-2    # Alex fixed: why isn't this also > 1.01?
        println("TOTAL FLUX NOT CONSERVED (should be 1): ", out_in_ratio)
        checkuConservation = false
    end

    if abs(output_flux) > 1e-2 #0.00001    # too stringent! 1e-2 enough
        checkUoutput = false
        println("FLUX AT OUTPUT TIP (shoud be 0): ", output_flux)
    end

    if checkUoutput && checkuConservation
        println("CONSERVATION CHECKS PASSED")
    else
        println("CONSERVATION CHECKS FAILED")
    end

end