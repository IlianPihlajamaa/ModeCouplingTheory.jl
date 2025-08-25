using Documenter
using ModeCouplingTheory

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "ModeCouplingTheory",
    pages = [
        "Introduction" => "index.md",
        "Equations and Solvers" => "Problems_and_Solvers.md",
        "Kernels" => "Kernels.md",
        "Mode-Coupling Theory" => "MCT.md",
        "Self-Consistent Generalized Langevin Equation Theory" => "SCGLET.md",
        "Active Mode-Coupling Theory" => "ActiveMCT.md",
        "Scope" => "Scope.md",
        "From Python" => "FromPython.md",
        "Internals" => "internals.md",
        "API Reference" => "API.md",
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