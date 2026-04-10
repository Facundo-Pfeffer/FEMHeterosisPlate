# Problem 1

A finite element program is to be written to solve a plate bending problem using a single element type: the **heterosis plate element** for a **shear-deformable isotropic elastic plate**.

This element is defined as follows:

- **Q8 interpolation** for the transverse displacement `u3`
- **Q9 interpolation** for the rotations `θ1` and `θ2`
- **3 × 3 quadrature** for the bending contribution to the stiffness
- **2 × 2 quadrature** for the shear contribution to the stiffness

The objective is to use this plate element to determine the **transverse deflection at point A** of the plate structure shown in the figure. The cut-out is centered.

## Geometry and material properties

- Overall plate dimensions: `500 mm × 300 mm`
- Centered rectangular cut-out: `250 mm × 180 mm`
- Thickness: `20 mm`
- Young's modulus: `E = 200 N/mm^2`
- Poisson's ratio: `ν = 0.25`
- Applied shear: `1 kN/mm`

## Boundary conditions

- The top and bottom outer edges are **free**
- The left and right outer edges are **clamped**
- The cut-out is centered, as indicated in the figure
- Point `A` is the point at which the transverse deflection must be computed

## Required task

Use the heterosis plate element to compute the transverse deflection at point `A`.

## Expected report contents

The report should include:

- A summary of the plate equations
- A summary of the finite element equations
- Test cases used during code development
- Convergence studies
- Comparison against baseline solutions, for example using FEAPpv