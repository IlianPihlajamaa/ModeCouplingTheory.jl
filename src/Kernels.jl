abstract type MemoryKernel end

include("Kernels/SchematicKernels.jl")
include("Kernels/ModeCouplingKernels.jl")
include("Kernels/MultiComponentKernels.jl")


"""
    evaluate_kernel!(out, kernel::MemoryKernel, F, t)

Evaluates the memory kernel in-place, overwriting the elements of the `out` variable. It may mutate the content of `kernel`.
"""
function evaluate_kernel! end


"""
    evaluate_kernel(kernel::MemoryKernel, F, t)
    
Evaluates the memory kernel out-place. It may mutate the content of `kernel`.

Returns
* `out` the kernel evaluated at (F, t)
"""
function evaluate_kernel end
