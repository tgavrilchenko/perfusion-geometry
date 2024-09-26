module YukBIE2D
# Yukawa layer potentials evaluation module in Julia.
# Barnett 7/25/22

include("utils.jl")
export
    di,
    unitcircle,
    starfish,
    interpmat1d,
    clencurt,
    chebydiffmat,
    perispec_deriv,
    perispec_deriv!

include("layerpots.jl")
export
    YukSLP,
    YukSLPeval,
    YukSLPmat,
    YukSLPmats,
    YukSLPmat_selfcrude,
    YukSLPdermat_selfcrude

end
