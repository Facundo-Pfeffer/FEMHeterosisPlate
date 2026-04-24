# Theory implemented by `FEMHeterosisPlate`

This document gives the continuum model, weak form, and element equations implemented in the repository. The notation is chosen to match the Python code as closely as possible.

The midsurface coordinates are denoted by \(x\) and \(y\). The thickness coordinate is denoted by \(z\), with \(z=0\) at the midsurface and \(-t/2\le z\le t/2\).

---

## 1. Plate kinematics

The model is a shear-deformable plate theory of Reissner--Mindlin type. A line initially normal to the midsurface remains straight after deformation, but it is not constrained to remain normal to the deformed midsurface. Therefore, transverse shear deformation is retained.

The independent midsurface fields are:

\[
w(x,y), \qquad \theta_x(x,y), \qquad \theta_y(x,y),
\]

where:

- \(w\) is the transverse deflection.
- \(\theta_x\) is the rotation variable associated with \(w_{,x}\).
- \(\theta_y\) is the rotation variable associated with \(w_{,y}\).

With the sign convention used by the code, the shear strains are:

\[
\gamma_{xz}=w_{,x}-\theta_x,
\qquad
\gamma_{yz}=w_{,y}-\theta_y.
\]

A compatible three-dimensional displacement field may be written in the form:

\[
u_x(x,y,z)=-z\,\theta_x(x,y),
\qquad
u_y(x,y,z)=-z\,\theta_y(x,y),
\qquad
u_z(x,y,z)=w(x,y).
\]

This field preserves straight transverse fibers and introduces independent rotations.

---

## 2. Strain measures

Using small-strain kinematics, the bending strain components vary linearly through the thickness:

\[
\varepsilon_{xx}=-z\,\theta_{x,x},
\qquad
\varepsilon_{yy}=-z\,\theta_{y,y},
\qquad
2\varepsilon_{xy}=-z(\theta_{x,y}+\theta_{y,x}).
\]

The transverse shear strains are constant through the thickness under the assumed kinematics:

\[
2\varepsilon_{xz}=w_{,x}-\theta_x,
\qquad
2\varepsilon_{yz}=w_{,y}-\theta_y.
\]

The finite element implementation works directly with the generalized bending and shear strain vectors:

\[
\boldsymbol{\kappa}
=
\begin{bmatrix}
\kappa_{xx}\\
\kappa_{yy}\\
2\kappa_{xy}
\end{bmatrix}
=
\begin{bmatrix}
\theta_{x,x}\\
\theta_{y,y}\\
\theta_{x,y}+\theta_{y,x}
\end{bmatrix},
\]

\[
\boldsymbol{\gamma}
=
\begin{bmatrix}
\gamma_{xz}\\
\gamma_{yz}
\end{bmatrix}
=
\begin{bmatrix}
w_{,x}-\theta_x\\
w_{,y}-\theta_y
\end{bmatrix}.
\]

The factor \(2\kappa_{xy}\) appears because the Voigt bending vector uses the engineering twisting component.

---

## 3. Stress resultants

The plate theory replaces the three-dimensional stresses by thickness-integrated resultants.

The bending and twisting moment resultants are:

\[
M_{\alpha\beta}
=
\int_{-t/2}^{t/2}
(-z)\sigma_{\alpha\beta}\,dz,
\qquad \alpha,\beta\in\{x,y\}.
\]

The transverse shear force resultants are:

\[
Q_\alpha
=
\int_{-t/2}^{t/2}
\sigma_{\alpha z}\,dz,
\qquad \alpha\in\{x,y\}.
\]

The stress-resultant vectors used in the code are:

\[
\boldsymbol{M}
=
\begin{bmatrix}
M_{xx}\\
M_{yy}\\
M_{xy}
\end{bmatrix},
\qquad
\boldsymbol{Q}
=
\begin{bmatrix}
Q_x\\
Q_y
\end{bmatrix}.
\]

---

## 4. Constitutive equations

The material is homogeneous, isotropic, and linearly elastic. The elastic constants are:

\[
E=\text{Young's modulus},
\qquad
\nu=\text{Poisson's ratio},
\qquad
G=\frac{E}{2(1+\nu)}.
\]

The in-plane response is treated under plane-stress assumptions. After integration through the thickness, the bending law becomes:

\[
\boldsymbol{M}
=
\boldsymbol{D}_b\boldsymbol{\kappa},
\]

where:

\[
\boldsymbol{D}_b
=
D
\begin{bmatrix}
1 & \nu & 0\\
\nu & 1 & 0\\
0 & 0 & \dfrac{1-\nu}{2}
\end{bmatrix},
\qquad
D=\frac{Et^3}{12(1-\nu^2)}.
\]

The transverse shear law is:

\[
\boldsymbol{Q}
=
\boldsymbol{D}_s\boldsymbol{\gamma},
\]

where:

\[
\boldsymbol{D}_s
=
k_sGt
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}.
\]

Here \(k_s\) is the transverse shear correction factor. The default code value is \(k_s=5/6\).

---

## 5. Strong form

Let \(p(x,y)\) denote transverse load per unit area, positive in the positive \(z\)-direction. The equilibrium equations are:

\[
Q_{x,x}+Q_{y,y}+p=0,
\]

\[
M_{xx,x}+M_{xy,y}+Q_x=0,
\]

\[
M_{xy,x}+M_{yy,y}+Q_y=0.
\]

In compact index notation:

\[
Q_{\alpha,\alpha}+p=0,
\]

\[
M_{\alpha\beta,\beta}+Q_\alpha=0.
\]

---

## 6. Boundary conditions

The essential boundary conditions prescribe the primary fields:

\[
w=\bar{w},
\qquad
\theta_x=\bar{\theta}_x,
\qquad
\theta_y=\bar{\theta}_y.
\]

Typical cases are:

- Clamped edge: \(w=0,\theta_x=0,\theta_y=0\).
- Simply supported edge in the implemented displacement setting: \(w=0\), with moment conditions left as natural where appropriate.
- Free edge: no essential boundary condition on that edge, so the corresponding natural terms are zero unless explicit loads are applied.

The theoretical natural boundary data are transverse shear traction and boundary moment resultants:

\[
Q_n = Q_\alpha n_\alpha,
\qquad
\bar{M}_\alpha=M_{\alpha\beta}n_\beta.
\]

The current implementation exposes transverse surface and edge loading. A separate boundary moment load object is not implemented in the current public interface.

---

## 7. Weak form

Let \(\delta w,\delta\theta_x,\delta\theta_y\) be admissible virtual fields satisfying the homogeneous form of the essential boundary conditions. The virtual curvature and shear fields are:

\[
\delta\boldsymbol{\kappa}
=
\begin{bmatrix}
\delta\theta_{x,x}\\
\delta\theta_{y,y}\\
\delta\theta_{x,y}+\delta\theta_{y,x}
\end{bmatrix},
\qquad
\delta\boldsymbol{\gamma}
=
\begin{bmatrix}
\delta w_{,x}-\delta\theta_x\\
\delta w_{,y}-\delta\theta_y
\end{bmatrix}.
\]

The displacement weak form is:

\[
\int_\Omega
\delta\boldsymbol{\kappa}^{T}
\boldsymbol{D}_b
\boldsymbol{\kappa}
\,d\Omega
+
\int_\Omega
\delta\boldsymbol{\gamma}^{T}
\boldsymbol{D}_s
\boldsymbol{\gamma}
\,d\Omega
=
\int_\Omega
\delta w\,p\,d\Omega
+
\int_{\Gamma_Q}
\delta w\,\bar{Q}\,d\Gamma
+
\int_{\Gamma_M}
\delta\boldsymbol{\theta}^{T}\bar{\boldsymbol{M}}\,d\Gamma.
\]

Only first derivatives of \(w,\theta_x,\theta_y\) appear. Therefore, \(C^0\)-continuous finite element fields are sufficient.

---

## 8. Current heterosis discretization

The current element implementation uses:

- Q8 serendipity interpolation for \(w\).
- Q9 Lagrange interpolation for \(\theta_x,\theta_y\).
- Q8 geometry mapping.
- Selective integration: \(3\times 3\) for bending and \(2\times 2\) for shear.

The element fields are:

\[
w^h(\xi,\eta)
=
\sum_{a=1}^{8}
N_a^{(8)}(\xi,\eta)w_a,
\]

\[
\theta_x^h(\xi,\eta)
=
\sum_{a=1}^{9}
N_a^{(9)}(\xi,\eta)\theta_{x,a},
\qquad
\theta_y^h(\xi,\eta)
=
\sum_{a=1}^{9}
N_a^{(9)}(\xi,\eta)\theta_{y,a}.
\]

The local element vector is:

\[
\boldsymbol{d}_e
=
\begin{bmatrix}
w_1 & \cdots & w_8 &
\theta_{x1} & \theta_{y1} &
\cdots &
\theta_{x9} & \theta_{y9}
\end{bmatrix}^{T}
\in\mathbb{R}^{26}.
\]

Thus, the current implementation has:

\[
8 + 2(9)=26
\]

local degrees of freedom per element.

---

## 9. Parent-to-physical mapping

The parent coordinates are:

\[
(\xi,\eta)\in[-1,1]\times[-1,1].
\]

The geometry map is:

\[
\boldsymbol{x}(\xi,\eta)
=
\sum_{a=1}^{8}
N_a^{(8)}(\xi,\eta)\boldsymbol{x}_a.
\]

The Jacobian is:

\[
\boldsymbol{J}
=
\begin{bmatrix}
x_{,\xi} & x_{,\eta}\\
y_{,\xi} & y_{,\eta}
\end{bmatrix}.
\]

Shape-function gradients are mapped with:

\[
\begin{bmatrix}
N_{,x}\\
N_{,y}
\end{bmatrix}
=
\boldsymbol{J}^{-T}
\begin{bmatrix}
N_{,\xi}\\
N_{,\eta}
\end{bmatrix}.
\]

The implementation requires \(\det \boldsymbol{J}>0\) at all area quadrature points.

---

## 10. Discrete bending matrix

The bending strain vector depends only on the rotational degrees of freedom. For a Q9 rotation node \(a\), the contributions are:

\[
\kappa_{xx}\leftarrow N^{(9)}_{a,x}\theta_{x,a},
\]

\[
\kappa_{yy}\leftarrow N^{(9)}_{a,y}\theta_{y,a},
\]

\[
2\kappa_{xy}\leftarrow
N^{(9)}_{a,y}\theta_{x,a}
+
N^{(9)}_{a,x}\theta_{y,a}.
\]

This defines the bending matrix \(\boldsymbol{B}_b\) such that:

\[
\boldsymbol{\kappa}^h=\boldsymbol{B}_b\boldsymbol{d}_e.
\]

---

## 11. Discrete shear matrix

The shear strain vector uses both the Q8 deflection field and the Q9 rotation field:

\[
\gamma_{xz}^h
=
\sum_{a=1}^{8} N_{a,x}^{(8)}w_a
-
\sum_{a=1}^{9} N_a^{(9)}\theta_{x,a},
\]

\[
\gamma_{yz}^h
=
\sum_{a=1}^{8} N_{a,y}^{(8)}w_a
-
\sum_{a=1}^{9} N_a^{(9)}\theta_{y,a}.
\]

This defines the shear matrix \(\boldsymbol{B}_s\) such that:

\[
\boldsymbol{\gamma}^h=\boldsymbol{B}_s\boldsymbol{d}_e.
\]

---

## 12. Element stiffness matrix

The element stiffness is:

\[
\boldsymbol{K}_e
=
\boldsymbol{K}_e^{(b)}
+
\boldsymbol{K}_e^{(s)},
\]

with:

\[
\boldsymbol{K}_e^{(b)}
=
\int_{\Omega_e}
\boldsymbol{B}_b^T
\boldsymbol{D}_b
\boldsymbol{B}_b
\,d\Omega,
\]

\[
\boldsymbol{K}_e^{(s)}
=
\int_{\Omega_e}
\boldsymbol{B}_s^T
\boldsymbol{D}_s
\boldsymbol{B}_s
\,d\Omega.
\]

The numerical approximation used by the code is:

\[
\boldsymbol{K}_e^{(b)}
\approx
\sum_{g=1}^{9}
\boldsymbol{B}_b^T(\xi_g,\eta_g)
\boldsymbol{D}_b
\boldsymbol{B}_b(\xi_g,\eta_g)
\det\boldsymbol{J}(\xi_g,\eta_g)
w_g,
\]

\[
\boldsymbol{K}_e^{(s)}
\approx
\sum_{g=1}^{4}
\boldsymbol{B}_s^T(\xi_g,\eta_g)
\boldsymbol{D}_s
\boldsymbol{B}_s(\xi_g,\eta_g)
\det\boldsymbol{J}(\xi_g,\eta_g)
w_g.
\]

---

## 13. Load vectors

A transverse surface load contributes only to the \(w\)-degrees of freedom:

\[
\boldsymbol{f}_e^{(p)}
=
\int_{\Omega_e}
\boldsymbol{N}_w^T p\,d\Omega.
\]

A transverse edge traction contributes only to the edge \(w\)-degrees of freedom:

\[
\boldsymbol{f}_e^{(q)}
=
\int_{\Gamma_e}
\boldsymbol{N}_{w,\Gamma}^T q\,d\Gamma.
\]

The current edge load uses a quadratic one-dimensional interpolation along each Q8 element edge.

---

## 14. Global system

After finite element assembly:

\[
\boldsymbol{K}\boldsymbol{u}
=
\boldsymbol{F}.
\]

The global unknown vector is ordered as:

\[
\boldsymbol{u}
=
\begin{bmatrix}
\text{all }w\text{ DOFs}\\
\text{all }\theta_x,\theta_y\text{ DOF pairs}
\end{bmatrix}.
\]

Essential boundary conditions are enforced by partitioning the system into free and constrained degrees of freedom:

\[
\boldsymbol{K}_{ff}\boldsymbol{u}_f
=
\boldsymbol{F}_f
-
\boldsymbol{K}_{fc}\boldsymbol{u}_c.
\]

The reduced sparse system is solved with SciPy's sparse direct solver.

---

## 15. Locking and selective integration

As the plate becomes thin, the Kirchhoff constraint is approached:

\[
\boldsymbol{\gamma}
=
\nabla w-\boldsymbol{\theta}
\rightarrow
\boldsymbol{0}.
\]

A displacement-based element can become artificially stiff if the discrete spaces cannot represent this constraint without over-constraining the deformation. This effect is known as shear locking.

The implementation uses selective integration to reduce the excessive stiffness contribution from the transverse shear term:

- the bending contribution is integrated with a higher-order rule,
- the shear contribution is integrated with a lower-order rule.

This is a standard finite element strategy for shear-deformable plates and shells, and it is part of the numerical behavior tested by the patch-test harness.

---

## 16. Comment on the uploaded Q9/Q8 theory convention

The uploaded theory text describes a heterosis element with Q9 interpolation for \(w\) and Q8 interpolation for rotations. The current repository implements Q8 interpolation for \(w\) and Q9 interpolation for rotations. This document follows the repository implementation.

If the project objective is the classical Q9-\(w\), Q8-\(\theta\) version, then the changes are not merely documentary. The element local vector, `HeterosisMesh`, `bending_B_matrix`, `shear_B_matrix`, load vectors, tests, and validation baselines must all be checked and revised consistently.
