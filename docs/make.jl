using Documenter
using ModeCouplingTheory

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "ModeCouplingTheory",
    pages = [
        "Introduction" => "index.md",
        "Problems and Solvers" => "Problems_and_Solvers.md",
        "Kernels" => "Kernels.md",
        "Scope" => "Scope.md",
        "From Python" => "FromPython.md",
        "API Reference" => "API.md",
        "Internals" => "internals.md",
     ],
     format = Documenter.HTML(prettyurls = false)
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/IlianPihlajamaa/ModeCouplingTheory.jl.git",
    devbranch = "main"
)