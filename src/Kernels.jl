abstract type MemoryKernel end

include("Kernels/SchematicKernels.jl")
include("Kernels/ModeCouplingKernels.jl")
include("Kernels/MultiComponentKernels.jl")


"""
    evaluate_kernel!(out, kernel::MemoryKernel, F, t)

Evaluates the memory kernel in-place, overwriting the elements of the `out` variable. It may mutate the content of `kernel`.
"""
function evaluate_kernel!(out, kernel::MemoryKernel, F, t)
end


"""
    evaluate_kernel(kernel::MemoryKernel, F, t)
    
Evaluates the memory kernel out-place. It may mutate the content of `kernel`.

Returns
* `out` the kernel evaluated at (F, t)
"""
function evaluate_kernel(kernel::MemoryKernel, F, t)
    @show typeof(kernel)
    @show typeof(F)
    @show typeof(t)
    error("The evaluate_kernel function is not defined for these argument types. This means that you are probably trying to call `evaluate_kernel(kernel, F, t)` with a kernel type for which this function has no specific method, or you are supplying it with an unsupported F(t=0) type.")
end