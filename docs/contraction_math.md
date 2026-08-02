# Contraction analysis of the damped double pendulum on T²

The mathematical details behind Example 5
([`double_pendulum_contraction/`](../src/contraction_examples/double_pendulum_contraction/)):
the state-dependent mechanical metric, the exact cross-term identity, the
certificates, and the honest approximations.

## 1. Dynamics

Absolute angles $q = (\theta_1, \theta_2) \in T^2$, $\Delta := \theta_1 - \theta_2$:

$$\mathcal{L} = \tfrac12 (m_1{+}m_2) l_1^2 \dot\theta_1^2 + \tfrac12 m_2 l_2^2 \dot\theta_2^2
+ m_2 l_1 l_2 \dot\theta_1 \dot\theta_2 \cos\Delta
+ (m_1{+}m_2) g l_1 \cos\theta_1 + m_2 g l_2 \cos\theta_2$$

Euler–Lagrange with viscous joint damping $D = \mathrm{diag}(d_1, d_2)$:

$$M(q)\ddot q + C(q,\dot q)\dot q + g(q) + D\dot q = 0$$

$$M(q) = \begin{bmatrix} a & b(q) \\ b(q) & c \end{bmatrix},\quad
a = (m_1{+}m_2)l_1^2,\ b(q) = m_2 l_1 l_2 \cos\Delta,\ c = m_2 l_2^2$$

$$C(q,\dot q) = m_2 l_1 l_2 \sin\Delta \begin{bmatrix} 0 & \dot\theta_2 \\ -\dot\theta_1 & 0 \end{bmatrix},
\qquad g(q) = \begin{bmatrix} (m_1{+}m_2) g l_1 \sin\theta_1 \\ m_2 g l_2 \sin\theta_2 \end{bmatrix}$$

$C$ is the Christoffel choice, so $\dot M - 2C$ is skew-symmetric. The
stiffness $K(q) = \partial g/\partial q$ is diagonal; $K_0 = K(0)$. State
$x = (q, \dot q)$, flow $\dot x = f(x)$, Jacobian $J(x) = \partial f/\partial x$.
Default parameters $m_i = 1$, $l_i = 1$, $g = 2$, $D = I$ give
$M_0 = \begin{bmatrix}2&1\\1&1\end{bmatrix}$, $K_0 = \mathrm{diag}(4, 2)$.

## 2. Contraction condition for a state-dependent metric

Virtual displacements obey $\delta\dot x = J(x)\,\delta x$. For
$V = \delta x^\top P(x)\,\delta x$:

$$\frac{d}{dt} V = \delta x^\top \big[\, \dot P + P J + J^\top P \,\big]\, \delta x,
\qquad \dot P = \sum_k \frac{\partial P}{\partial x_k} f_k(x).$$

So contraction at rate $\lambda$ on a region $K$ is

$$R(x) := \dot P(x) + P(x) J(x) + J(x)^\top P(x) \ \preceq\ -2\lambda P(x) \quad \text{on } K .$$

Our $P$ depends only on $q$, hence $\dot P = \sum_k (\partial P/\partial q_k)\dot q_k$
is linear in $\dot q$ and **vanishes identically on the $\dot q = 0$ slice** —
the painted torus needs no $\dot P$ term.

## 3. The energy metric only semi-contracts

Along the linearization $M_0 \delta\ddot q = -D\delta\dot q - K_0\delta q$, the
$\delta$-energy $V_E = \delta\dot q^\top M_0 \delta\dot q + \delta q^\top K_0 \delta q$ gives

$$\dot V_E = -2\,\delta\dot q^\top D\,\delta\dot q - 2\,\delta\dot q^\top K_0 \delta q + 2\,\delta q^\top K_0 \delta\dot q
= -2\,\delta\dot q^\top D\,\delta\dot q \ \preceq\ 0,$$

with the cross terms cancelling by symmetry of $K_0$. Dissipation acts only
through $\delta\dot q$: for $P_E = \mathrm{blkdiag}(K_0, M_0)$ the spectrum of
$P_E A + A^\top P_E$ is $\{0, 0, -2, -2\}$ — semi-contraction, the 2-DOF
version of the stalling Euclidean circle of the mass-spring-damper example.

## 4. Cross-term repair: the exact constant-chart identity

$$P_0 = \begin{bmatrix} K_0 + \varepsilon D & \varepsilon M_0 \\ \varepsilon M_0 & M_0 \end{bmatrix},
\qquad A = \begin{bmatrix} 0 & I \\ -M_0^{-1} K_0 & -M_0^{-1} D \end{bmatrix}.$$

One block multiplication:

$$P_0 A = \begin{bmatrix} -\varepsilon K_0 & K_0 + \varepsilon D - \varepsilon D \\ -K_0 & \varepsilon M_0 - D \end{bmatrix}
= \begin{bmatrix} -\varepsilon K_0 & K_0 \\ -K_0 & \varepsilon M_0 - D \end{bmatrix}.$$

The off-diagonal blocks $K_0$ and $-K_0$ are exactly skew, so symmetrization
annihilates them:

$$\boxed{\,P_0 A + A^\top P_0 = -2\,\mathrm{blkdiag}\big(\varepsilon K_0,\ D - \varepsilon M_0\big)\,}$$

which is $\prec 0$ iff $K_0 \succ 0$ and $D - \varepsilon M_0 \succ 0$.
Equivalently, $V = \tfrac12\delta\dot q^\top M_0\delta\dot q
+ \tfrac12\delta q^\top (K_0{+}\varepsilon D)\delta q
+ \varepsilon\,\delta q^\top M_0 \delta\dot q$ has
$\dot V = -\delta\dot q^\top (D - \varepsilon M_0)\delta\dot q - \varepsilon\,\delta q^\top K_0 \delta q$.
Admissibility: $P_0 \succ 0 \iff M_0 \succ 0$ and (Schur)
$K_0 + \varepsilon D - \varepsilon^2 M_0 \succ 0$; strict contraction needs
$\varepsilon < 1/\lambda_{\max}(D^{-1} M_0) = 2/(3+\sqrt5) \approx 0.382$ for the defaults.
The identity is verified to machine precision in the test runs.

## 5. The state-dependent metric

$$P(q) = \begin{bmatrix} K_0 + \varepsilon D & \varepsilon M(q) \\ \varepsilon M(q) & M(q) \end{bmatrix},
\qquad
\dot P = \begin{bmatrix} 0 & \varepsilon \dot M \\ \varepsilon \dot M & \dot M \end{bmatrix},
\quad \dot M = -m_2 l_1 l_2 \sin\Delta\,(\dot\theta_1 - \dot\theta_2) \begin{bmatrix} 0&1\\1&0 \end{bmatrix}.$$

**Positive definiteness, rigorously.** $P(q)$ depends *affinely* on the scalar
$b(q) \in [-m_2 l_1 l_2,\ +m_2 l_1 l_2]$, and the cone of positive-definite
matrices is convex — so positive definiteness at the two endpoints
($\Delta = 0$ and $\Delta = \pi$) implies it for every $q$. That endpoint check
is exactly what `MechanicalMetric.__post_init__` performs.

**Why $M(q)$ rather than $M_0$.** In the velocity–velocity block of $R(x)$
the metric-rate term $\dot M$ meets the Coriolis content of
$\partial \ddot q/\partial \dot q$, and the mechanical identity
$\dot q^\top(\dot M - 2C)\dot q = 0$ (skew-symmetry) lets them cancel instead
of leaving $\dot M$ as an unpaired indefinite disturbance — the same
cancellation that powers energy arguments. Freezing $M_0$ discards the
$\dot M$ term but leaves the $C$-terms and the $M(q) - M_0$ mismatch
uncancelled. With the $\varepsilon$ cross-blocks the full $4\times4$ algebra
no longer collapses to a closed form, so beyond the linearization $R(x) \prec 0$
is certified numerically — and the payoff is measured: at the default fan the
state-dependent metric passes with margin $-0.31$ while the constant variant
fails for every admissible $\varepsilon$.

## 6. From $R \prec 0$ to decay of distances

Transport a curve $\gamma_0(s)$, $s \in [0,1]$, joining two states, by the flow:
$\gamma_t = \varphi_t \circ \gamma_0$. Its metric length is

$$L(t) = \int_0^1 \sqrt{w^\top P(\gamma_t(s))\, w}\; ds, \qquad w = \partial \gamma_t/\partial s,$$

and each $w$ is a genuine virtual displacement along the trajectory through
$\gamma_0(s)$. If $R \preceq -2\lambda P$ at every point of the transported
curves, every integrand decays at rate $\lambda$, so
$L(t) \le e^{-\lambda t} L(0)$ and the induced distance obeys the same bound.
This is why the certificate is evaluated **along the straight segments between
trajectory pairs** — precisely the curves whose lengths the right panel plots
— with guaranteed rate

$$\lambda = \tfrac12\, \min_{x \in \text{segments}}\ \lambda_{\min}\!\big({-R(x)},\ P(x)\big)
\quad \text{(generalized eigenvalue)},$$

$0.064$ for the default fan, against the observed average
$-\max \mathrm{Re}\,\mathrm{eig}(A) = 0.22$ — the same average-vs-bound gap
seen in the 1-DOF examples.

**Stated approximations** (both deliberate and visible in the code): for a
state-dependent metric the true Riemannian distance minimizes over geodesics,
not straight segments — the segment length used here is a computable upper
bound, exact in the constant-metric case; and the certificate samples a grid
of times × pairs × interpolation points rather than a continuum.

## 7. The painted slice, and why contraction must be local

At $\dot q = 0$: $C(q, 0) = 0$, $\dot P = 0$,
$J_{22} = -M^{-1}(q) D$, $J_{21} = -\partial\,[M^{-1} g]/\partial q$, so the
painted quantity $\lambda_{\max}(P(q) J + J^\top P(q))$ is the exact
contraction condition on that slice (blue island $=$ contracting; dashed line
$=$ zero contour). Globally, no metric can contract all of $T^2$: contraction
forces all trajectories to converge to a single equilibrium, and the double
pendulum has four — $(0,0)$ stable and $(0,\pi), (\pi,0), (\pi,\pi)$ unstable
(the open circles on the red part of the torus). The island is necessarily an
island.

## 8. Numbers for the defaults ($m{=}1$, $l{=}1$, $g{=}2$, $D{=}I$, $\varepsilon{=}0.25$)

| Quantity | Value |
|---|---|
| $\mathrm{eig}(A)$ | $-1.28 \pm 2.25i$, $-0.22 \pm 1.07i$ (Hurwitz) |
| Energy metric ($\varepsilon{=}0$): $\mathrm{eig}(P_E A + A^\top P_E)$ | $\{0, 0, -2, -2\}$ — semi only |
| Identity $P_0 A + A^\top P_0 + 2\,\mathrm{blkdiag}(\varepsilon K_0, D{-}\varepsilon M_0)$ | $0$ to machine precision |
| Segment certificate, fan $\pm 0.18$, $P(q)$ | $\max \lambda_{\max}(R) = -0.31 \prec 0$, rate $0.064$ |
| Same fan, constant $P_0$ (any admissible $\varepsilon$) | fails ($+0.08 \ldots +0.13$) |
| Certified fan boundary | $\approx 0.20$ rad for $P(q)$ vs $\approx 0.15$–$0.18$ for $P_0$ |
