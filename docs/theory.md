# Theory implemented by `FEMHeterosisPlate`

This document gives the continuum model, weak form, and element equations implemented in the repository. The notation is chosen to match the Python code as closely as possible.

The midsurface coordinates are denoted by $x$ and $y$. The thickness coordinate is denoted by $z$, with $z=0$ at the midsurface and $-t/2 \le z \le t/2$.

---

## 1. Plate kinematics

The model is a shear-deformable plate theory of Reissner--Mindlin type. A line initially normal to the midsurface remains straight after deformation, but it is not constrained to remain normal to the deformed midsurface. Therefore, transverse shear deformation is retained.

The independent midsurface fields are:

$$
w(x,y), \qquad \theta_x(x,y), \qquad \theta_y(x,y),
$$

where:

- $w$ is the transverse deflection.
- $\theta_x$ is the rotation variable associated with $w_{,x}$.
- $\theta_y$ is the rotation variable associated with $w_{,y}$.

With the sign convention used by the code, the shear strains are:

$$
\begin{aligned}
\gamma_{xz} &= w_{,x}-\theta_x, \\
\gamma_{yz} &= w_{,y}-\theta_y.
\end{aligned}
$$

A compatible three-dimensional displacement field may be written as:

$$
\begin{aligned}
u_x(x,y,z) &= -z\,\theta_x(x,y), \\
u_y(x,y,z) &= -z\,\theta_y(x,y), \\
u_z(x,y,z) &= w(x,y).
\end{aligned}
$$

This field preserves straight transverse fibers and introduces independent rotations.

---

## 2. Strain measures

Using small-strain kinematics, the bending strain components vary linearly through the thickness:

$$
\begin{aligned}
\varepsilon_{xx} &= -z\,\theta_{x,x}, \\
\varepsilon_{yy} &= -z\,\theta_{y,y}, \\
2\varepsilon_{xy} &= -z\left(\theta_{x,y}+\theta_{y,x}\right).
\end{aligned}
$$

The transverse shear strains are constant through the thickness under the assumed kinematics:

$$
\begin{aligned}
2\varepsilon_{xz} &= w_{,x}-\theta_x, \\
2\varepsilon_{yz} &= w_{,y}-\theta_y.
\end{aligned}
$$

The finite element implementation works directly with the generalized bending and shear strain vectors:

$$
\kappa
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
$$

$$
\gamma
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
$$

The factor $2\kappa_{xy}$ appears because the Voigt bending vector uses the engineering twisting component.

---

## 3. Stress resultants

The plate theory replaces the three-dimensional stresses by thickness-integrated resultants.

The bending and twisting moment resultants are:

$$
M_{\alpha\beta}
=
\int_{-t/2}^{t/2}
(-z)\sigma_{\alpha\beta}\,dz,
\qquad \alpha,\beta\in\{x,y\}.
$$

The transverse shear force resultants are:

$$
Q_\alpha
=
\int_{-t/2}^{t/2}
\sigma_{\alpha z}\,dz,
\qquad \alpha\in\{x,y\}.
$$

The stress-resultant vectors used in the code are:

$$
\mathbf{M}
=
\begin{bmatrix}
M_{xx}\\
M_{yy}\\
M_{xy}
\end{bmatrix},
\qquad
\mathbf{Q}
=
\begin{bmatrix}
Q_x\\
Q_y
\end{bmatrix}.
$$

---

## 4. Constitutive equations

The material is homogeneous, isotropic, and linearly elastic. The elastic constants are:

$$
E=\text{Young's modulus},
\qquad
\nu=\text{Poisson's ratio},
\qquad
G=\frac{E}{2(1+\nu)}.
$$

The in-plane response is treated under plane-stress assumptions. After integration through the thickness, the bending law becomes:

$$
\mathbf{M}=\mathbf{D}_b\kappa,
$$

where:

$$
\mathbf{D}_b
=
D
\begin{bmatrix}
1 & \nu & 0\\
\nu & 1 & 0\\
0 & 0 & \dfrac{1-\nu}{2}
\end{bmatrix},
\qquad
D=\frac{Et^3}{12(1-\nu^2)}.
$$

The transverse shear law is:

$$
\mathbf{Q}=\mathbf{D}_s\gamma,
$$

where:

$$
\mathbf{D}_s
=
k_sGt
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}.
$$

Here $k_s$ is the transverse shear correction factor. The default code value is $k_s=5/6$.

---

## 5. Strong form

Let $p(x,y)$ denote transverse load per unit area, positive in the positive $z$-direction. The equilibrium equations are:

$$
Q_{x,x}+Q_{y,y}+p=0,
$$

$$
M_{xx,x}+M_{xy,y}+Q_x=0,
$$

$$
M_{xy,x}+M_{yy,y}+Q_y=0.
$$

In compact index notation:

$$
Q_{\alpha,\alpha}+p=0,
$$

$$
M_{\alpha\beta,\beta}+Q_\alpha=0.
$$

---

## 6. Boundary conditions

The essential boundary conditions prescribe the primary fields:

$$
\begin{aligned}
w &= \bar{w}, \\
\theta_x &= \bar{\theta}_x, \\
\theta_y &= \bar{\theta}_y.
\end{aligned}
$$

Typical cases are:

- Clamped edge: $w=0$, $\theta_x=0$, and $\theta_y=0$.
- Simply supported edge in the implemented displacement setting: $w=0$, with moment conditions left as natural where appropriate.
- Free edge: no essential boundary condition on that edge, so the corresponding natural terms are zero unless explicit loads are applied.

The theoretical natural boundary data are transverse shear traction and boundary moment resultants:

$$
Q_n = Q_\alpha n_\alpha,
\qquad
\bar{M}_\alpha=M_{\alpha\beta}n_\beta.
$$

The current implementation exposes transverse surface and edge loading. A separate boundary moment load object is not implemented in the current public interface.

---

## 7. Weak form

Let $\delta w$, $\delta\theta_x$, and $\delta\theta_y$ be admissible virtual fields satisfying the homogeneous form of the essential boundary conditions. The virtual curvature and shear fields are:

$$
\delta\kappa
=
\begin{bmatrix}
\delta\theta_{x,x}\\
\delta\theta_{y,y}\\
\delta\theta_{x,y}+\delta\theta_{y,x}
\end{bmatrix},
\qquad
\delta\gamma
=
\begin{bmatrix}
\delta w_{,x}-\delta\theta_x\\
\delta w_{,y}-\delta\theta_y
\end{bmatrix}.
$$

The displacement weak form is:

$$
\begin{aligned}
&\int_\Omega
\delta\kappa^{T}
\mathbf{D}_b
\kappa
\,d\Omega
+\int_\Omega
\delta\gamma^{T}
\mathbf{D}_s
\gamma
\,d\Omega \\
&\qquad =
\int_\Omega
\delta w\,p\,d\Omega
+\int_{\Gamma_Q}
\delta w\,\bar{Q}\,d\Gamma
+\int_{\Gamma_M}
\delta\theta^{T}\bar{\mathbf{M}}\,d\Gamma.
\end{aligned}
$$

Only first derivatives of $w$, $\theta_x$, and $\theta_y$ appear. Therefore, $C^0$-continuous finite element fields are sufficient.

---

## 8. Current heterosis discretization

The current element implementation uses:

- Q8 serendipity interpolation for $w$.
- Q9 Lagrange interpolation for $\theta_x$ and $\theta_y$.
- Q8 geometry mapping.
- Selective integration: $3\times 3$ for bending and $2\times 2$ for shear.

The element fields are:

$$
w^h(\xi,\eta)
=
\sum_{a=1}^{8}
N_a^{(8)}(\xi,\eta)w_a,
$$

$$
\begin{aligned}
\theta_x^h(\xi,\eta)
&=
\sum_{a=1}^{9}
N_a^{(9)}(\xi,\eta)\theta_{x,a}, \\
\theta_y^h(\xi,\eta)
&=
\sum_{a=1}^{9}
N_a^{(9)}(\xi,\eta)\theta_{y,a}.
\end{aligned}
$$

The local element vector is:

$$
\mathbf{d}_e
=
\begin{bmatrix}
w_1 & \cdots & w_8 &
\theta_{x1} & \theta_{y1} &
\cdots &
\theta_{x9} & \theta_{y9}
\end{bmatrix}^{T}
\in\mathbb{R}^{26}.
$$

Thus, the current implementation has:

$$
8 + 2(9)=26
$$

local degrees of freedom per element.

---

## 9. Parent-to-physical mapping

The parent coordinates are:

$$
(\xi,\eta)\in[-1,1]\times[-1,1].
$$

The geometry map is:

$$
\mathbf{x}(\xi,\eta)
=
\sum_{a=1}^{8}
N_a^{(8)}(\xi,\eta)\mathbf{x}_a.
$$

The Jacobian is:

$$
\mathbf{J}
=
\begin{bmatrix}
x_{,\xi} & x_{,\eta}\\
y_{,\xi} & y_{,\eta}
\end{bmatrix}.
$$

Shape-function gradients are mapped with:

$$
\begin{bmatrix}
N_{,x}\\
N_{,y}
\end{bmatrix}
=
\mathbf{J}^{-T}
\begin{bmatrix}
N_{,\xi}\\
N_{,\eta}
\end{bmatrix}.
$$

The implementation requires $\det \mathbf{J}>0$ at all area quadrature points.

---

## 10. Discrete bending matrix

The bending strain vector depends only on the rotational degrees of freedom. For a Q9 rotation node $a$, the contributions are:

$$
\kappa_{xx}\leftarrow N^{(9)}_{a,x}\theta_{x,a},
$$

$$
\kappa_{yy}\leftarrow N^{(9)}_{a,y}\theta_{y,a},
$$

$$
2\kappa_{xy}\leftarrow
N^{(9)}_{a,y}\theta_{x,a}
+N^{(9)}_{a,x}\theta_{y,a}.
$$

This defines the bending matrix $\mathbf{B}_b$ such that:

$$
\kappa^h=\mathbf{B}_b\mathbf{d}_e.
$$

---

## 11. Discrete shear matrix

The shear strain vector uses both the Q8 deflection field and the Q9 rotation field:

$$
\begin{aligned}
\gamma_{xz}^h
&=
\sum_{a=1}^{8} N_{a,x}^{(8)}w_a
-\sum_{a=1}^{9} N_a^{(9)}\theta_{x,a}, \\
\gamma_{yz}^h
&=
\sum_{a=1}^{8} N_{a,y}^{(8)}w_a
-\sum_{a=1}^{9} N_a^{(9)}\theta_{y,a}.
\end{aligned}
$$

This defines the shear matrix $\mathbf{B}_s$ such that:

$$
\gamma^h=\mathbf{B}_s\mathbf{d}_e.
$$

---

## 12. Element stiffness matrix

The element stiffness is:

$$
\mathbf{K}_e
=
\mathbf{K}_e^{(b)}
+\mathbf{K}_e^{(s)},
$$

with:

$$
\mathbf{K}_e^{(b)}
=
\int_{\Omega_e}
\mathbf{B}_b^T
\mathbf{D}_b
\mathbf{B}_b
\,d\Omega,
$$

$$
\mathbf{K}_e^{(s)}
=
\int_{\Omega_e}
\mathbf{B}_s^T
\mathbf{D}_s
\mathbf{B}_s
\,d\Omega.
$$

The numerical approximation used by the code is:

$$
\mathbf{K}_e^{(b)}
\approx
\sum_{g=1}^{9}
\mathbf{B}_b^T(\xi_g,\eta_g)
\mathbf{D}_b
\mathbf{B}_b(\xi_g,\eta_g)
\det\mathbf{J}(\xi_g,\eta_g)
w_g,
$$

$$
\mathbf{K}_e^{(s)}
\approx
\sum_{g=1}^{4}
\mathbf{B}_s^T(\xi_g,\eta_g)
\mathbf{D}_s
\mathbf{B}_s(\xi_g,\eta_g)
\det\mathbf{J}(\xi_g,\eta_g)
w_g.
$$

---

## 13. Load vectors

A transverse surface load contributes only to the $w$-degrees of freedom:

$$
\mathbf{f}_e^{(p)}
=
\int_{\Omega_e}
\mathbf{N}_w^T p\,d\Omega.
$$

A transverse edge traction contributes only to the edge $w$-degrees of freedom:

$$
\mathbf{f}_e^{(q)}
=
\int_{\Gamma_e}
\mathbf{N}_{w,\Gamma}^T q\,d\Gamma.
$$

The current edge load uses a quadratic one-dimensional interpolation along each Q8 element edge.

---

## 14. Global system

After finite element assembly:

$$
\mathbf{K}\mathbf{u}=\mathbf{F}.
$$

The global unknown vector is ordered as:

$$
\mathbf{u}
=
\begin{bmatrix}
\text{all }w\text{ degrees of freedom}\\
\text{all }\theta_x,\theta_y\text{ degree-of-freedom pairs}
\end{bmatrix}.
$$

Essential boundary conditions are enforced by partitioning the system into free and constrained degrees of freedom:

$$
\mathbf{K}_{ff}\mathbf{u}_f
=
\mathbf{F}_f
-\mathbf{K}_{fc}\mathbf{u}_c.
$$

The reduced sparse system is solved with SciPy's sparse direct solver.

---

## 15. Locking and selective integration

As the plate becomes thin, the Kirchhoff constraint is approached:

$$
\gamma
=
\nabla w-\theta
\rightarrow
\mathbf{0}.
$$

A displacement-based element can become artificially stiff if the discrete spaces cannot represent this constraint without over-constraining the deformation. This effect is known as shear locking.

The implementation uses selective integration to reduce the excessive stiffness contribution from the transverse shear term:

- The bending contribution is integrated with a higher-order rule.
- The shear contribution is integrated with a lower-order rule.

This is a standard finite element strategy for shear-deformable plates and shells, and it is part of the numerical behavior tested by the patch-test harness.
