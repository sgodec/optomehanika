using Random:default_rng, seed!
using LaTeXStrings
using CairoMakie
using ProgressMeter
using LinearAlgebra

MT = Makie.MathTeXEngine
mt_fonts_dir = joinpath(dirname(pathof(MT)), "..", "assets", "fonts", "NewComputerModern")

set_theme!(fonts = (
regular = joinpath(mt_fonts_dir, "NewCM10-Regular.otf"),
bold = joinpath(mt_fonts_dir, "NewCM10-Bold.otf")
) )
