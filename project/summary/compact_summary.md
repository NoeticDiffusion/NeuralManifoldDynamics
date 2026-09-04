# NDT / MNDM compact formula summary

```text
PURPOSE:
- machine-oriented compression of the formula layer in `articles/compendium/Noetic Diffusion Theory Compendium.typ`
- prefer symbolic reuse over prose
- use as canonical notation map, not as publication text

CORE SYMBOLS:
- X_t in M                      = latent noetic state on manifold M
- x_t = [m_t, d_t, e_t]^T       = 3D chart state
- x_t^(9)                       = 9D stratified chart state
- Y_t                           = measured neural data
- g                             = measurement map
- epsilon_t                     = observation noise
- G_t(x)                        = effective geometry / conductance tensor
- Phi(x,t)                      = effective potential
- phi_meta(t)                   = slow meta-regulatory field
- sigma(t), Sigma(t)            = diffusion/noise amplitude
- W_t                           = Brownian motion
- J(x,t)                        = Meta-Noetic Jacobian (MNJ)
- S = (J + J^T)/2               = symmetric part
- Omega = (J - J^T)/2           = antisymmetric / rotational part
- A_t, b_t, Q_t                 = local linear reachability parameters
- Sigma_t^(h)                   = h-step reachability covariance

IMPLEMENTED MAIN-REPOSITORY DEPENDENCY:
- J_hat -> Phi (time-ordered finite-time response, opt-in)
- residual_covariance_proxy + Phi -> W_Q (only for an explicitly admissible
  one-step transition covariance)

AXIS SEMANTICS:
- m = metastability / mobility / rhythmic coordination
- d = deviation from optimal integration-segregation balance
- e = entropy / entropic energy / complexity pressure
- legacy (E,R,D)-style triplets are treated as monotone-aligned precursors to (m,d,e)

CHARTS:
- x_t = [m_t, d_t, e_t]^T in R^3
- x_t^(9) = [m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m]^T in R^9
- Y_t = g(X_t) + epsilon_t

==================================================
1. CORE NDT / MNDM LATENT DYNAMICS
==================================================

MINIMAL MODEL:
- dX_t = -grad F_min(X_t) dt + sigma(t) dW_t
- F_min(X) = e(X) - m(X)
- sigma(t) = sigma_min + sigma_0 (1 - r(t))
- r(t) in [0,1] = rhythmic control; high r => lower noise / stronger denoising

EXTENDED GEOMETRIC MODEL:
- dX_t = -G_t(X_t) grad Phi(X_t,t) dt + sigma(t) dW_t
- Phi(X,t) = alpha(t)e(X) - beta(t)m(X) + gamma(t)d(X) - delta(t)phi_meta(t)
- drift: b(x,t) = -G_t(x) grad Phi(x,t)

GENERATOR:
- L psi(x) = b(x,t) . grad psi(x) + (1/2) Tr(a(x,t) nabla^2 psi(x))
- a(x,t) = Sigma(t) Sigma(t)^T

EMPIRICAL COARSE DRIFT/DIFFUSION:
- b_hat(x) approx (1/Delta t) E[X_(k+1) - X_k | X_k approx x]
- a_hat(x) approx (1/Delta t) Cov[X_(k+1) - X_k | X_k approx x]

WELL-POSEDNESS ENVELOPE:
- (M,g) complete Riemannian manifold
- Phi(.,t) at least C^2 in state, C^1 in time
- G_t(x) symmetric positive definite, Lipschitz in x
- sigma(t) piecewise continuous, bounded away from 0

IDENTIFIABILITY WARNING:
- data constrain local law / generator more directly than a unique decomposition into (G_t, Phi, sigma)
- gauge freedoms: chart change, time reparametrization, additive constants in Phi

==================================================
2. MEASUREMENT LAYER: MNPS / STRATIFIED MNPS
==================================================

9D DEFINITIONS:
- m_a = SR * alpha_attn_mean
- m_e = SR * (max(z_L(w), 0))_W
- SC(w) = (sum_f f P(f,w)) / (sum_f P(f,w))
- m_o = std_W(SC) / avg_W(SC)
- H(t_h,w) = exp(-t_h L(w))
- d_n = 1 - norm(avg_W(Tr(H(t_h,w)) / |V|))
- d_l = avg_W(C_i(w))
- d_s = std_u(proj(b_u))    or    d_s = avg_W(1 - rho(RDM(s), RDM(b(w))))
- e_e = (1/S) sum_(s=1)^S SampEn_s
- e_s = H_perm(phi, m_p, tau)
- e_m = z((P_beta + P_gamma) / (P_alpha + P_theta))    or    e_m = z(avg_W(E_g(w)))

RECOMPOSITION TO 3D:
- m = robust_mean(m_a, m_e, m_o)
- d = robust_mean(d_n, d_l, d_s)
- e = robust_mean(e_e, e_s, e_m)

WEIGHTED ALTERNATIVE:
- m = w_a m_a + w_e m_e + w_o m_o
- d = w_n d_n + w_l d_l + w_s d_s
- e = w_ee e_e + w_es e_s + w_em e_m
- weights nonnegative, sum to 1 within family

LATENT -> MEASUREMENT MONOTONE BRIDGES:
- e_e(X_t) approx h_e(E(X_t))
- e_s(X_t) approx h_s(r(X_t))
- d_n(X_t) approx norm(Tr(G_t(X_t)))
- d_l(X_t) approx h_l(kappa(X_t))
- m_o(t) approx Var_W(SC(sigma(t)))
- m_a(t) approx h_a(precision pulses aligned to theta)

CURRENT NMD v2 CONTRACT:
- X_t_export = [m_t, d_t, e_t]
- X_t_export^(9) = [m_a, m_e, m_o, d_n, d_l, d_s, e_e, e_s, e_m]
- x_t = P_fixed x_t^(9)

ACTIVE EEG-STYLE POLICY EXAMPLE:
- m = 0.62 m_a + 0.55 m_e + 0.45 m_o
- d = 0.50 d_n + 0.82 d_l + 0.28 d_s
- e = 0.85 e_e + 0.62 e_s + 0.03 e_m

CONTRACT NOTE:
- P_fixed is a release-fixed auditable projection, not a claim of unique ontology
- downstream MNJ/reachability claims inherit this contract

==================================================
3. CYTODENDRITIC / ACCESSIBILITY BRIDGE
==================================================

MICROSTATE:
- Z_t = { x_b(t), s_i(t), M_b(t), E_b(t), P_b(t) }_(b,i)
  where:
  - x_b = fast branch integration state
  - s_i = local spine/synapse access state
  - M_b = slow structural accessibility
  - E_b = eligibility trace
  - P_b = local capture/transport/metabolic support

ACCESS FACTOR:
- A_b(t) = A_b^f(x_b(t), s(t), C(t)) * A_b^s(M_b(t))

FAST DYNAMICS:
- xdot_b = A_b(t) F_b(x, I, s)
- sdot_i = G_i(s, x, M)

SLOW STRUCTURAL UPDATE:
- Mdot_b(t) = eta E_b(t) sigma(delta(t) - theta_delta) (1 - M_b(t)/M_max)
              - lambda_M M_b(t)
              + sqrt(2 T_eff) xi_b(t)

TRACE-SPECIFIC RECALL SUPPORT:
- R_mu(t) = sum_b a_(mu b) A_b(t) x_b(t)
- yhat_mu(t) = Phi_read(R_mu(t) - theta_mu)

MESOSCOPIC EXPORT:
- Psi: Z_t -> x_t^(mem)
- xdot_t^(mem) = f_mem(x_t^(mem), t) + epsilon_t
- J_t^(mem) = partial f_mem / partial x (x_t^(mem), t)
- Sigma_t^(h+1) = A_t Sigma_t^(h) A_t^T + Q_t

MICRO -> MACRO METRIC / POTENTIAL COUPLING:
- G_MNPS(X,Z) approx P^T G_access(Psi(Z)) P
- Phi(X,t,Z) = Phi_0(X,t) + Phi_access(X,t,Psi(Z))

BLOCK METRIC / BLOCK JACOBIAN:
- G_block(r,s,alpha,beta;X,Z)
    approx sum_a sum_b P(r <- (a,alpha)) G_access(Psi(Z))_(a,b) P(s <- (b,beta))
- H(u,s,gamma,beta) = partial^2 Phi / [partial x^(u)_gamma partial x^(s)_beta]
- J_block(r,s,alpha,beta;x,t)
    approx -sum_u sum_gamma G_block(r,u,alpha,gamma;x,t) H(u,s,gamma,beta;x,t)

BOUNDARY:
- this bridge is a speculative forward model from micro-accessibility to MNPS/MNJ/reachability
- not required for macro-level inference; not yet directly validated as a unique biological substrate

==================================================
4. META-NOETIC JACOBIAN (MNJ)
==================================================

FLOW:
- f(x,t) = xdot(t)

MNJ DEFINITION:
- J_ij(x,t) = partial f_i(x,t) / partial x_j = partial xdot_i / partial x_j

3D MATRIX:
- J(x,t) =
  [ dmdot/dm  dmdot/dd  dmdot/de
    dddot/dm  dddot/dd  dddot/de
    dedot/dm  dedot/dd  dedot/de ]

CANONICAL DECOMPOSITION:
- S = (1/2)(J + J^T)
- Omega = (1/2)(J - J^T)

DERIVED INVARIANTS:
- Tr(J) = div f
- J v_k = lambda_k v_k
- Jdot = dJ/dt

EMPIRICAL LOCAL LINEAR ESTIMATION:
- xdot approx A(x - xbar) + b
- J_hat = A
- discrete/continuous relation:
  - J_d = exp(J_c Delta t)
  - J_c approx (J_d - I) / Delta t     for small Delta t

RELATION TO POTENTIAL CURVATURE:
- if f(x,t) = -G_t(x) grad Phi(x,t), then
- J(x,t) = -G_t(x) nabla^2 Phi(x,t) - R(x,t)
- R_(i,j)(x,t) = sum_k [partial_j G_(i,k)(x,t)] [partial_k Phi(x,t)]
- common approximation:
  - J(x,t) approx -G_t(x) nabla^2 Phi(x,t)
  - valid only when ||R|| is small relative to ||G_t nabla^2 Phi||

9D / FAMILY-BLOCK JACOBIAN:
- J^(9) =
  [ J_(m,m)  J_(m,d)  J_(m,e)
    J_(d,m)  J_(d,d)  J_(d,e)
    J_(e,m)  J_(e,d)  J_(e,e) ]
- each J_(out,in) in R^(3x3)

==================================================
5. REACHABILITY CONES / LOCAL CAPACITY GEOMETRY
==================================================

LOCAL STOCHASTIC LINEARIZATION:
- x_(t+1) = A_t x_t + b_t + epsilon_t
- epsilon_t ~ (0, Q_t)

REACHABILITY COVARIANCE RECURSION:
- Sigma_t^(0) = 0
- Sigma_t^(h+1) = A_t Sigma_t^(h) A_t^T + Q_t

ELLIPSOIDAL REACHABLE SET:
- E_t^(h)(alpha) = { z | z^T (Sigma_t^(h))^(-1) z <= alpha }

PRIMARY CONE METRICS:
- let lambda_1 >= ... >= lambda_d = eigenvalues of Sigma_t^(h)
- volume proxy: Vol_t^(h) = (1/2) log det(Sigma_t^(h))
- anisotropy / condition: kappa_t^(h) = lambda_1 / (lambda_d + epsilon)
- effective dimension:
  - d_eff,t^(h) = (sum_i lambda_i)^2 / (sum_i lambda_i^2)

ROTATION PROXIES:
- if A_t = W Sigma V^T (SVD), define U_t = W V^T
- rotation proxy from A:
  - rho_t^(A) = || (U_t - U_t^T)/2 ||_F
- optional Jacobian rotation:
  - rho_t = || (J_t - J_t^T)/2 ||_F

PERSISTENCE / RECOVERY:
- using tube metric V_t = tube_log_det(t)
- tau_rec(t) = inf_(Delta t > 0) { Delta t : V_(t+Delta t) >= V_rec }

SECONDARY CONE METRICS:
- p_i = (lambda_i + epsilon) / (sum_j lambda_j + d epsilon)
- H_dir = -sum_i p_i log(p_i)
- Htilde_dir = H_dir / (log(d) + epsilon)
- transport ratio:
  - r_vol(t) = exp(tube_log_det(t) - tube_log_det(t-1))

INTERPRETATION DISCIPLINE:
- reachability != occupancy
- reachability != traversability
- large Q_t can inflate cones without implying richer controllable geometry
- horizon dependence and neighborhood leakage are validity conditions

==================================================
6. C / TI / P REGIME GEOMETRY
==================================================

REGIME DESCRIPTOR:
- R_t^(CTIP) = (C_t, TI_t, P_t)

MINIMAL OPERATIONAL READING:
- C_t  = C(speed, tube_log_det, d_eff, rho)
- TI_t = TI(Q-ratio, CaptureGate, MDR)
- P_t  = P(ACI, CFCA, asymmetry)

SEMANTICS:
- C  = capacity collapse / reduced local accessibility
- TI = transport-innovation balance: innovation-dominated vs transport-captured
- P  = pathological lock-in / canalized asymmetric deformation

IMPORTANT DISSOCIATION:
- low capacity alone is underdetermined
- state reports should specify whether observed reduction is mainly C, TI, P, or mixture

==================================================
7. TRAVERSABILITY / EXTENSIONS
==================================================

TRAVERSABILITY INDEX:
- T = v * (1 - A_norm)
- A_norm = (A - A_min) / (A_max - A_min)

READING:
- low T can arise from low speed, high canalization, or both
- T is a compact diagnostic, not a replacement for MNJ or reachability geometry

JUMP-DIFFUSION EXTENSION:
- dX_t = -nabla_G Phi(X_t,t) dt + sigma(t) dW_t
         + integral_(R^d) kappa(X_(t^-), z) N_tilde(dt,dz)

MODALITY BOUNDARY:
- EEG / iEEG: first-order + second-order layers often estimable
- fMRI: first-order geometry more reliable than MNJ/reachability; second-order claims need stronger validity checks

==================================================
8. DEPENDENCY CHAIN
==================================================

PIPELINE:
1. latent process:
   X_t evolves under MNDM
2. measurement:
   Y_t = g(X_t) + epsilon_t
3. charting:
   Y_t -> x_t or x_t^(9)
4. projection contract:
   x_t = P_fixed x_t^(9)   (release-fixed operational layer)
5. local flow:
   f(x,t) = xdot(t)
6. local Jacobian:
   J = partial f / partial x
7. short-horizon capacity geometry:
   Sigma_t^(h+1) = A_t Sigma_t^(h) A_t^T + Q_t
8. regime classification:
   R_t^(CTIP) = (C_t, TI_t, P_t)

MICRO-ACCESSIBILITY OPTIONAL AUGMENT:
- Z_t -> Psi(Z_t) -> G_access / Phi_access -> altered MNJ / reachability / CTIP summaries

==================================================
9. SAFE INFERENCE RULES
==================================================

- do not treat measurement coordinates as the latent manifold itself
- do not treat P_fixed coefficients as universal constants
- do not treat J approx -G nabla^2 Phi as exact unless metric-variation remainder is small
- do not equate cone size with occupancy, entropy, speed, or consciousness directly
- do not collapse C/TI/P into a single scalar story
- do not treat the cytodendritic bridge as established biology; it is a forward model
- generator / local law is more identifiable than any unique latent mechanistic decomposition

ONE-LINE GLOBAL SUMMARY:
- NDT = latent stochastic dynamics on a geometry-shaped manifold, observed through a versioned measurement chart, differentiated into local flow/Jacobian structure, propagated into short-horizon reachability geometry, and summarized at regime level by capacity collapse (C), transport-innovation balance (TI), and pathological lock-in (P).

==================================================
10. EXPLORATORY STATUS: ESB -> TRAJECTORY GEOMETRY
==================================================

10A. ESB / PRL / HYSTERESIS  (article-local clinical layer; not core compendium layer)

STATUS:
- article-local and downstream of MNPS / reachability / MNJ-style exports
- useful for perturbational clinical analysis
- not part of the canonical NDT compendium backbone
- current repo status:
  - ESB-2 = primary empirical metric in the photic ESB article
  - PRL = secondary / negative-bounded result
  - hysteresis / recovery = exploratory due state-matching limitations

ESB-2 DEFINITIONS:
- Delta C_i(f) = C_i(f) - C_i(rest)
- Delta R_i(f) = R_i(f) - R_i(rest)
- h_C(f) = median_HC[Delta C_i(f)]
- h_R(f) = median_HC[Delta R_i(f)]
- s_C(f) = MAD_HC[Delta C_i(f)]
- s_R(f) = MAD_HC[Delta R_i(f)]
- ESB-2_i(f)
    = (1/sqrt(2)) *
      sqrt(
        ((Delta C_i(f) - h_C(f)) / (s_C(f) + epsilon))^2 +
        ((Delta R_i(f) - h_R(f)) / (s_R(f) + epsilon))^2
      )

ESB-2 READING:
- scalar magnitude of deviation from healthy rest-to-photic template
- low ESB-2 => close to healthy evoked template
- high ESB-2 => abnormal evoked deformation
- scalar only; directionality must be recovered from axis-wise Delta C / Delta R analysis

CURRENT PRIMARY ESB SPECIFICATION IN REPO:
- capacity = parietal-occipital tube_d_eff_median
- guidance = global rotational_power_median
- baseline = matched ds004504 rest

PRL:
- PRL_i = 1 - corr(subject_ESB2_profile_i, healthy_median_ESB2_profile)
- profile usually taken across PHOTO 5 / 10 / 15 / 20 Hz in the current article
- intended meaning: loss of healthy frequency-response shape / rhythmic selectivity

HYSTERESIS / RECOVERY:
- H_i = distance(post_stimulus_residual_i, baseline_i)
- current article-local residual term:
  - h_rest_raw = distance(post-photic residual geometry, ds004504 resting baseline)
- state-matching warning:
  - interpretable recovery requires comparable baseline and residual states
  - eyes-closed rest vs post-photic residual is bounded / confounded, not a clean recovery theorem

ESB-3:
- ESB-3 approx ESB-2 + normalized recovery term
- conceptually: evoked brittleness + residual non-return
- current status: bounded / exploratory, not the strongest defended claim

SAFE USE:
- do not fold ESB into canonical NDT formalism as if it were a foundational layer
- treat ESB-2 as an internally stabilized perturbational deviation score on a frozen article-local surface
- treat PRL and recovery as secondary unless stronger frequency support and state matching exist

10B. TRAJECTORY / REPERTOIRE GEOMETRY  (exploratory proposal; put after ESB)

STATUS:
- explicitly exploratory / proposal-stage
- non-Jacobian by design
- intended to sit above exported MNPS 3D/9D trajectories
- not part of current standard NeuralManifoldDynamics H5 contract
- proposed role:
  - MNJ => local deformation of flow field
  - reachability => admissible nearby futures
  - trajectory geometry => realized temporal grammar of traversal

BASIC OBJECT:
- z_t in R^k, with k = 3 or 9

10B-1. META-NOETIC REPERTOIRE FLOW (MNRF)
- discretize:
  - s_t = q(z_t),  s_t in {1,...,N}
- transition matrix:
  - P_ij = Pr(s_(t+1)=j | s_t=i)
- stationary / occupancy mass:
  - pi_i

REPERTOIRE ENTROPY RATE:
- H_rate = -sum_i pi_i sum_j P_ij log(P_ij + epsilon)

IRREVERSIBILITY / FLUX:
- I = (1/2) sum_(i,j) |pi_i P_ij - pi_j P_ji|

ENTROPY PRODUCTION VARIANT:
- E_prod = sum_(i,j) pi_i P_ij log((pi_i P_ij + epsilon)/(pi_j P_ji + epsilon))

OPTIONAL AUXILIARY READOUTS:
- self-loop rate = sum_i pi_i P_ii
- return probability = Pr(s_(t+tau) = s_t)
- transition sparsity = sparsity(P)

10B-2. NOETIC ACTION / PATH COST
- Delta z_t = z_(t+1) - z_t
- Delta^2 z_t = z_(t+1) - 2 z_t + z_(t-1)
- ||u||_M^2 = u^T M u
- A_MNPS
    = sum_t [ (1/2) ||Delta z_t||_M^2 / Delta t^2
              + eta ||Delta^2 z_t||_M^2 / Delta t^4
              + lambda Phi(z_t) ] Delta t

POTENTIAL CHOICES:
- 3D:
  - Phi(z_t) = alpha e_t - beta m_t + gamma d_t
- 9D family-aggregate example:
  - mbar_t = w_m^T [m_a, m_e, m_o]
  - dbar_t = w_d^T [d_n, d_l, d_s]
  - ebar_t = w_e^T [e_e, e_s, e_m]
  - Phi(z_t^(9)) = alpha_e ebar_t - beta_m mbar_t + gamma_d dbar_t

LOCAL / GLOBAL ACTION SUMMARY:
- L_t = local action integrand / lagrangian
- NAS = median_t(L_t)

10B-3. CURVATURE / TORSION / TORTUOSITY
- v_t = (z_(t+1) - z_t) / Delta t
- a_t = (v_(t+1) - v_t) / Delta t
- j_t = (a_(t+1) - a_t) / Delta t

CURVATURE:
- kappa_t = sqrt(||v_t||^2 ||a_t||^2 - (v_t . a_t)^2) / (||v_t||^3 + epsilon)

TORTUOSITY:
- T_path = [sum_t ||z_(t+1) - z_t||] / (||z_T - z_1|| + epsilon)

TORSION (3D only):
- tau_t = det(v_t, a_t, j_t) / (||v_t x a_t||^2 + epsilon)

10B-4. HYSTERESIS AREA / PATH-MEMORY GEOMETRY
- planar loop area in MNPS plane (i,j):
  - H_ij = (1/2) |sum_t z_i(t) z_j(t+1) - z_j(t) z_i(t+1)|
- path hysteresis:
  - H_path = integral_0^1 ||z_down(s) - z_up(s)|| ds

READING:
- tests whether induction and recovery use same manifold corridor or a different return path

10B-5. STRATIFIED FAMILY TEMPORAL COUPLING
- C_ab(tau) = corr(Delta abar_t, bbar_(t-tau)),  where a,b in {m,d,e}
- A_ab = C_ab(tau) - C_ba(tau)
- intended meaning: temporal family coupling / drive asymmetry without Jacobian estimation

RECOMMENDED FIRST ENDPOINTS:
- H_rate
- I
- A_MNPS
- kappa_median
- T_path

SAFE USE:
- keep trajectory geometry under exploratory / proposal status unless implemented and validated
- do not represent it as part of the current standard NDT / NMD contract
- do not conflate realized path grammar with local Jacobian deformation or with reachability capacity
```
