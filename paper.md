---
title: 'ModeCouplingTheory.jl: A solver for mode-coupling-theory-like integro-differential equations'
tags:
  - julia
  - python
  - dynamics
  - glasses
  - liquids
  - integral equations
  - differential equations
authors:
  - name: Ilian Pihlajamaa
    orcid: 0000-0003-3779-4281
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Corentin C.L. laudicina
    orcid: 0009-0000-1888-2417
    affiliation: 1
  - name: Thomas Voigtmann
    orcid: 0000-0002-1261-9295
    affiliation: "2,3"
  - name: Liesbeth M.C. Janssen
    orcid: 0000-0001-5283-1330
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Soft Matter and Biological Physics, Department of Applied Physics, Eindhoven University of Technology, P.O. Box 513, 5600 MB Eindhoven, Netherlands
   index: 1
 - name: Institut für Materialphysik im Weltraum, Deutsches Zentrum für Luft- und Raumfahrt (DLR), Köln, 51170, Germany
   index: 2
 - name: Heinrich Heine University, Universitätsstraße 1, Düsseldorf, 40225, Germany
   index: 3
date: 14 July 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The mode-coupling theory of the glass transition is one of the most succesful theories for predicting the dynamics of the glass transition to date.
In the regime where it is applicable, the theory gives remarkably accurate predictions for the drastic degree in which the structural relaxation time scale slows down upon supercooling.
In particular, the theory predicts that density correlation functions evolve according to a generalized Langevin equation, which is an integro-differential equation, of which the integration kernel couples "relaxation modes". Because these equations are difficult to solve numerically, due to their long-livedness, specialized algorithms have been developed to tackle this issue. `ModeCouplingTheory.jl` is a package that implements these algorithms, including a number of convenience features that make it simple to solve the complex equations involved even for those not well-versed in the theoretical and numerical background traditionally required.  

# Statement of need

The mode-coupling theory of the glass transition (MCT) is at the forefront of the study of the glass transition [@gotze2009complex; @leutheusser1984dynamical; @bengtzelius1984dynamics; @das2004mode; @janssen2018mode; @reichman2005mode] and has been applied to an abundance of different systems and scenarios in the past forty years. See for example versions of MCT including multiple particle species[@gotze2003effect; @weysser2010structural; @franosch2002completely; @luo2022many], confinement [@lang2010glass; @krakoviack2007mode], self-propelling particles [@berthier2013non; @feng2017mode; @liluashvili2017mode; @debets2023mode; @reichert2021mode], and molecular particles [@theis2000test; @chong1998mode; @schilling1997mode; @winkler2000molecular; @chong2000mode] to name a few. Before the conception of the software this paper is based on, there was no open-source integrator for MCT-like equations available. This meant that each time an improvement, extension or application to the theory was employed, a new in-house integrator needed to be developed or adapted. `ModeCouplingTheory.jl` aims at providing an open source and tested implementation of the scheme introduced by @fuchs1991comments, that is performant and easy to extend to new systems, yet simple to call interactively from dynamic languages such as `Python`. As of writing, it has been used for several scientific works [@winter2023deep; @pihlajamaa2023unveiling; @laudicina2023competing].

The main equation that this package aims to solve is given by

$$\alpha \ddot{F}(t) + \beta \dot{F}(t) + \gamma F(t) + \delta + \int_0^t d\tau K(t-\tau)\dot{F}(\tau) = 0$$

in which $\alpha$, $\beta$, $\gamma$, and $\delta$ are coefficients (possibly dependent on time), $K(t) = K(F(t), t)$ is the memory kernel, and $F(t)$ is the function that is being solved for. Memory kernels for solving the most commonly encountered models, such as standard (multicomponent) MCT, including tagged-particle correlators, and mean squared displacements are implemented, and it is straightforward to implement custom memory kernels. Internally, the software completely separates the memory kernel, the type of equation, and the solver, each of which can be independenty extended by a user.  

The documentation details the features of this software, which among others include

1.  Generality: the code was developed with generality in mind. For example, the code works for types between which product operation is defined among $\alpha$, $\beta$, $\gamma$, $K$ on the left and $F$ on the right, returning something of the same type as $F$ and $\delta$. This implies that the code works for functions $F$ that are scalar valued (schematic models), as well as those that are vectors of floating point numbers (standard MCT), and vectors with elements of different types. The latter could include, for example, numbers with measurement errors, dual numbers, and immutable matrices. 

2. Speed: the code is developed for performance. The solver allocates little memory and uses BLAS implementations for linear algebra where applicable [@lawson1979basic]. The memory kernels of the single component and multi-component MCT as well as their tagged variants are implemented using Bengtzelius' trick, yielding dramatic algorithmic speed-up compared to more naive implementations.

3. Ease of use: solving the equations of standard MCT essentially takes four lines of code. While writted in `Julia`, the code can straightforwardly be called from `Python` and other languages. Convenience functions 

4.  Measurement errors: by leveraging the generality of the software it is straightforward to do standard error propagation through the solver. 

5.  Automatic differentiation: similarly, the use of dual numbers allow for forward-mode automatic differentiation. This allows, for example, the use of nearal networks as surrogates for memory kernels or efficient methods for inverse problems.

6.  Non-ergodicity parameters: there is built-in functionality for finding the steady state solutions of the equation. 

# Example Use

To solve the standard MCT equations in three dimensions, one may run the following code. See the documentation for more explanations.

```julia
using ModeCouplingTheory
# the wave vector grid
Nk = 100; kmax = 40.0; dk = kmax/Nk; k_array = range(dk/2, kmax-dk/2, length=Nk);
# a very bad approximation of the structure factor
Sk = @. 1 - cos(k_array)*exp(-k_array) 

# physical parameters and coefficients
kBT = 1.0; m = 1.0; ρ = 1.5
∂F0 = zeros(Nk); α = 0.0; β = 1.0; γ = @. k_array^2*kBT/(m*Sk); δ = 0.0

kernel = ModeCouplingKernel(ρ, kBT, m, k_array, Sk)
equation = MemoryEquation(α, β, γ, δ, Sk, ∂F0, kernel)
sol = solve(equation);

# plot the solution for several values of k
using Plots
p = plot(xlabel="log10(t)", ylabel="F(k,t)", ylims=(0,1), xlims=(-4, 1))
for ik = [7, 18, 25, 39]
    Fk = get_F(sol, :, ik)
    plot!(p, log10.(get_t(sol)), Fk/Sₖ[ik], label="k = $(k_array[ik])", lw=3)
end
p
```
![The code above yields this figure, which shows the intermediate scattering function, obtained with MCT, as a function of time for different values of $k$.\label{fig:example}](paperfig.pdf)

# Acknowledgements

We acknowledge valuable discussions from Daniele Coslovich. This work has been financially supported by theDutch Research Council (NWO) through a Vidi grant (IP,CCLL, and LMCJ) and START-UP grant (LMCJ)

# References