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
+
N^{(9)}_{a,x}\theta_{y,a}.
$$

This defines the bending matrix $\mathbf{B}_b$ such that:

$$
\mathbf{\kappa}^h=\mathbf{B}_b\mathbf{d}_e.
$$

---

## 11. Discrete shear matrix

The shear strain vector uses both the Q8 deflection field and the Q9 rotation field:

$$
\gamma_{xz}^h
=
\sum_{a=1}^{8} N_{a,x}^{(8)}w_a
-
\sum_{a=1}^{9} N_a^{(9)}\theta_{x,a},
$$

$$
\gamma_{yz}^h
=
\sum_{a=1}^{8} N_{a,y}^{(8)}w_a
-
\sum_{a=1}^{9} N_a^{(9)}\theta_{y,a}.
$$

This defines the shear matrix $\mathbf{B}_s$ such that:

$$
\mathbf{\gamma}^h=\mathbf{B}_s\mathbf{d}_e.
$$

---

## 12. Element stiffness matrix

The element stiffness is:

$$
\mathbf{K}_e
=
\mathbf{K}_e^{(b)}
+
\mathbf{K}_e^{(s)},
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
\mathbf{K}\mathbf{u}
=
\mathbf{F}.
$$

The global unknown vector is ordered as:

$$
\mathbf{u}
=
\begin{bmatrix}
\text{all }w\text{ DOFs}\\
\text{all }\theta_x,\theta_y\text{ DOF pairs}
\end{bmatrix}.
$$

Essential boundary conditions are enforced by partitioning the system into free and constrained degrees of freedom:

$$
\mathbf{K}_{ff}\mathbf{u}_f
=
\mathbf{F}_f
-
\mathbf{K}_{fc}\mathbf{u}_c.
$$

The reduced sparse system is solved with SciPy's sparse direct solver.

---

## 15. Locking and selective integration

As the plate becomes thin, the Kirchhoff constraint is approached:

$$
\mathbf{\gamma}
=
\nabla w-\mathbf{\theta}
\rightarrow
\mathbf{0}.
$$

A displacement-based element can become artificially stiff if the discrete spaces cannot represent this constraint without over-constraining the deformation. This effect is known as shear locking.

The implementation uses selective integration to reduce the excessive stiffness contribution from the transverse shear term:

- the bending contribution is integrated with a higher-order rule,
- the shear contribution is integrated with a lower-order rule.

This is a standard finite element strategy for shear-deformable plates and shells, and it is part of the numerical behavior tested by the patch-test harness.