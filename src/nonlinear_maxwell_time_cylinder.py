# Colab/Python version of the nonlinear time-dependent Maxwell cylinder model.
# Install first in Colab:
# !pip -q install pyvista vtk imageio imageio-ffmpeg
# !apt-get -qq update
# !apt-get -qq install -y xvfb > /dev/null
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.special import jv, jvp, jn_zeros, jnp_zeros
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
import imageio.v2 as imageio
import pyvista as pv
from IPython.display import Image, display

# Headless rendering for Colab.
pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass

rng = np.random.default_rng(1)

# ================================================================
# 1. PARAMETERS
# ================================================================
R = 1.0
H = 2.0
gamma = 1.0

t_end = 20.0
n_time = 121
t_eval = np.linspace(0.0, t_end, n_time)

# 60 is quicker in Colab; change to 100 if you want.
n_animation_frames = 32
gif_fps = 4

n_basis = 7
n_eigen_curves = 5

m_max = 4
ell_max = 4
p_max = 3

# ================================================================
# 2. CYLINDRICAL MAXWELL MODES
#
# We use TWO divergence-free families.
#
# Toroidal:
#   T = curl(psi e_z),
#   psi = J_m(alpha r/R) cos(m theta) cos(p pi z/H),
#   J_m(alpha)=0.
#
# Poloidal:
#   P = curl curl(psi e_z),
#   psi = J_m(beta r/R) cos(m theta) sin(p pi z/H),
#   J_m'(beta)=0, p>=1.
#
# The poloidal family has a nonzero z-component, so Figure 2 is
# genuinely three-dimensional instead of a stack of planar flows.
# ================================================================

def build_candidate_modes(R, H, m_max, ell_max, p_max):
    modes = []
    for m in range(m_max + 1):
        jzeros = jn_zeros(m, ell_max)
        jpzeros = jnp_zeros(m, ell_max)

        # Toroidal family
        for ell in range(1, ell_max + 1):
            alpha = float(jzeros[ell - 1])
            for p in range(0, p_max + 1):
                lam = (alpha / R) ** 2 + (p * np.pi / H) ** 2
                modes.append({
                    "family": "T",
                    "m": m,
                    "ell": ell,
                    "p": p,
                    "alpha": alpha,
                    "lambda": lam,
                })

        # Poloidal family
        for ell in range(1, ell_max + 1):
            beta = float(jpzeros[ell - 1])
            for p in range(1, p_max + 1):
                lam = (beta / R) ** 2 + (p * np.pi / H) ** 2
                modes.append({
                    "family": "P",
                    "m": m,
                    "ell": ell,
                    "p": p,
                    "alpha": beta,
                    "lambda": lam,
                })

    return sorted(modes, key=lambda d: d["lambda"])


def select_distinct_modes(candidates, n_basis, tol=1e-7):
    selected = []
    for cand in candidates:
        if all(abs(cand["lambda"] - old["lambda"]) > tol for old in selected):
            selected.append(cand)
        if len(selected) >= n_basis:
            break
    return selected


def cylindrical_components(mode, R, H, r, theta, z):
    """
    Return Cartesian components of one raw Maxwell mode.
    Inputs r, theta, z can be NumPy arrays of any common shape.
    """
    fam = mode["family"]
    m = int(mode["m"])
    p = int(mode["p"])
    alpha = float(mode["alpha"])

    kappa = alpha / R
    q = p * np.pi / H

    r = np.asarray(r)
    theta = np.asarray(theta)
    z = np.asarray(z)

    J = jv(m, kappa * r)
    Jp = jvp(m, kappa * r, 1)

    cm = np.cos(m * theta)
    sm = np.sin(m * theta)

    # Stable version of m*J_m(kappa r)/r near r=0.
    mJ_over_r = np.zeros_like(r, dtype=float)
    mask = r > 1e-12
    if m != 0:
        mJ_over_r[mask] = m * J[mask] / r[mask]
        if m == 1:
            mJ_over_r[~mask] = kappa / 2.0

    if fam == "T":
        # T = curl(psi e_z)
        zf = np.cos(q * z)

        Ur = -mJ_over_r * sm * zf
        Utheta = -kappa * Jp * cm * zf
        Uz = np.zeros_like(r)

    elif fam == "P":
        # P = curl curl(psi e_z)
        # psi uses sin(q z), hence p>=1.
        zc = np.cos(q * z)
        zs = np.sin(q * z)

        Ur = q * kappa * Jp * cm * zc
        Utheta = -q * mJ_over_r * sm * zc
        Uz = (kappa ** 2) * J * cm * zs

    else:
        raise ValueError("Unknown mode family")

    # Cylindrical -> Cartesian
    Ux = Ur * np.cos(theta) - Utheta * np.sin(theta)
    Uy = Ur * np.sin(theta) + Utheta * np.cos(theta)

    return Ux, Uy, Uz


def evaluate_mode_cartesian(mode, R, H, x, y, z):
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return cylindrical_components(mode, R, H, r, theta, z)


def evaluate_mode_set_cartesian(modes, R, H, x, y, z, order="C"):
    """
    Evaluate all raw modes at arbitrary Cartesian points.
    Returns BX, BY, BZ with shape (number_of_points, number_of_modes).
    """
    xf = np.asarray(x).ravel(order=order)
    yf = np.asarray(y).ravel(order=order)
    zf = np.asarray(z).ravel(order=order)

    P = xf.size
    M = len(modes)

    BX = np.zeros((P, M))
    BY = np.zeros((P, M))
    BZ = np.zeros((P, M))

    for j, mode in enumerate(modes):
        ux, uy, uz = evaluate_mode_cartesian(mode, R, H, xf, yf, zf)
        BX[:, j] = ux
        BY[:, j] = uy
        BZ[:, j] = uz

    return BX, BY, BZ


def combine_modes(BX, BY, BZ, coeff):
    return BX @ coeff, BY @ coeff, BZ @ coeff


# ================================================================
# 3. SELECT BASIS MODES
# ================================================================
candidates = build_candidate_modes(R, H, m_max, ell_max, p_max)
raw_modes = select_distinct_modes(candidates, n_basis)
n_basis = len(raw_modes)

print("Selected mixed toroidal/poloidal cylindrical basis:")
print(" j   fam   m  ell  p       lambda")
print("------------------------------------------")
for j, mode in enumerate(raw_modes, start=1):
    print(
        f"{j:2d}    {mode['family']:1s}    "
        f"{mode['m']:1d}   {mode['ell']:1d}   {mode['p']:1d}   "
        f"{mode['lambda']:10.6f}"
    )

# ================================================================
# 4. CYLINDRICAL QUADRATURE GRID
# ================================================================
nr = 19
ntheta = 28
nz = 16

dr = R / nr
dtheta = 2 * np.pi / ntheta
dz = H / nz

r = (np.arange(nr) + 0.5) * dr
theta = (np.arange(ntheta) + 0.5) * dtheta
z = (np.arange(nz) + 0.5) * dz

RR, TH, ZZ = np.meshgrid(r, theta, z, indexing="ij")
W = (RR * dr * dtheta * dz).ravel()
n_quad = W.size

RawX = np.zeros((n_quad, n_basis))
RawY = np.zeros((n_quad, n_basis))
RawZ = np.zeros((n_quad, n_basis))

for j, mode in enumerate(raw_modes):
    ux, uy, uz = cylindrical_components(mode, R, H, RR, TH, ZZ)
    RawX[:, j] = ux.ravel()
    RawY[:, j] = uy.ravel()
    RawZ[:, j] = uz.ravel()

# ================================================================
# 5. MASS AND CURL-CURL MATRICES
# ================================================================
Mraw = (
    RawX.T @ (W[:, None] * RawX)
    + RawY.T @ (W[:, None] * RawY)
    + RawZ.T @ (W[:, None] * RawZ)
)
Mraw = 0.5 * (Mraw + Mraw.T)

lambda_raw = np.array([m["lambda"] for m in raw_modes])

Kraw = 0.5 * (
    Mraw @ np.diag(lambda_raw)
    + np.diag(lambda_raw) @ Mraw
)

# Generalized self-adjoint eigenproblem.
lambda_linear, Vlin = eigh(Kraw, Mraw)

# Mraw-normalize to reduce accumulated quadrature error.
for j in range(n_basis):
    Vlin[:, j] /= np.sqrt(Vlin[:, j] @ Mraw @ Vlin[:, j])

PhiX = RawX @ Vlin
PhiY = RawY @ Vlin
PhiZ = RawZ @ Vlin

print("\nLinear curl-curl eigenvalues:")
print(lambda_linear)

# ================================================================
# 6. PRECOMPUTE QUARTIC TENSOR
#
# T[i,j,a,b] =
#   integral (phi_i.phi_j)(phi_a.phi_b)
#
# Then
# G_i(c) = sum_{j,a,b} T[i,j,a,b] c_j c_a c_b.
# ================================================================
S = np.empty((n_quad, n_basis * n_basis))

for a in range(n_basis):
    for b in range(n_basis):
        S[:, a * n_basis + b] = (
            PhiX[:, a] * PhiX[:, b]
            + PhiY[:, a] * PhiY[:, b]
            + PhiZ[:, a] * PhiZ[:, b]
        )

Cpair = S.T @ (W[:, None] * S)
Cpair = 0.5 * (Cpair + Cpair.T)
T4 = Cpair.reshape(n_basis, n_basis, n_basis, n_basis)


def nonlinear_G_J(c):
    # Cubic Galerkin vector
    G = np.einsum(
        "ijab,j,a,b->i",
        T4, c, c, c,
        optimize=True,
    )

    # D G(c)
    term1 = np.einsum(
        "ijab,a,b->ij",
        T4, c, c,
        optimize=True,
    )
    term2 = np.einsum(
        "iajb,a,b->ij",
        T4, c, c,
        optimize=True,
    )

    JG = term1 + 2.0 * term2
    return G, JG


# ================================================================
# 7. PHYSICAL TIME EVOLUTION
#
# q'' + Lambda q + gamma G(q) = 0
# ================================================================
q0 = np.zeros(n_basis)
q0[0] = 1.10

if n_basis >= 2:
    q0[1] = 0.32

if n_basis >= 4:
    q0[3] = 0.12

qdot0 = np.zeros(n_basis)

if n_basis >= 3:
    qdot0[2] = 0.22

Y0 = np.r_[q0, qdot0]


def dynamic_rhs(t, Y):
    q = Y[:n_basis]
    qdot = Y[n_basis:]

    G, _ = nonlinear_G_J(q)
    qddot = -lambda_linear * q - gamma * G

    return np.r_[qdot, qddot]


sol = solve_ivp(
    dynamic_rhs,
    (0.0, t_end),
    Y0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10,
    method="RK45",
)

if not sol.success:
    raise RuntimeError(sol.message)

t_solution = sol.t
Q = sol.y[:n_basis, :]
Qdot = sol.y[n_basis:, :]

# ================================================================
# 8. ENERGY DIAGNOSTIC
# ================================================================
energy = np.zeros(n_time)
N_of_t = np.zeros(n_time)

for k in range(n_time):
    q = Q[:, k]
    qdot = Qdot[:, k]

    G, _ = nonlinear_G_J(q)
    quartic = q @ G

    energy[k] = (
        0.5 * (qdot @ qdot)
        + 0.5 * np.sum(lambda_linear * q**2)
        + 0.25 * gamma * quartic
    )

    N_of_t[k] = q @ q

relative_energy_drift = (
    np.max(energy) - np.min(energy)
) / abs(energy[0])

print(f"\nRelative energy drift = {relative_energy_drift:.3e}")
print(
    "N(t)=||u(t)||^2 varies between "
    f"{N_of_t.min():.4f} and {N_of_t.max():.4f}"
)

# ================================================================
# 9. FROZEN-TIME LINEARISED SPECTRUM
#
# L(t) = diag(lambda_linear) + gamma DG(q(t)).
# ================================================================
n_eigen_curves = min(n_eigen_curves, n_basis)

lambda_frozen = np.full((n_eigen_curves, n_time), np.nan)
phi1_coeff = np.zeros((n_basis, n_time))

previous_mode = None
eig_tol = 1e-9

for k in range(n_time):
    q = Q[:, k]

    _, JG = nonlinear_G_J(q)

    Lfrozen = np.diag(lambda_linear) + gamma * JG
    Lfrozen = 0.5 * (Lfrozen + Lfrozen.T)

    vals, vecs = np.linalg.eigh(Lfrozen)

    positive = np.where(vals > eig_tol)[0]
    if positive.size == 0:
        raise RuntimeError(
            f"No positive frozen eigenvalue at t={t_solution[k]:.4f}"
        )

    vals_pos = vals[positive]
    vecs_pos = vecs[:, positive]

    nkeep = min(n_eigen_curves, len(vals_pos))
    lambda_frozen[:nkeep, k] = vals_pos[:nkeep]

    # Smallest positive frozen eigenvector.
    mode = vecs_pos[:, 0]
    mode /= np.linalg.norm(mode)

    if previous_mode is not None and np.dot(mode, previous_mode) < 0:
        mode = -mode

    phi1_coeff[:, k] = mode
    previous_mode = mode

# Instantaneous gaps, useful for checking separation.
gaps = np.diff(lambda_frozen, axis=0)

# ================================================================
# 10. VISUALIZATION GRIDS
# ================================================================

# ---------- PyVista 3D flow grid ----------
n_flow_xy = 25
n_flow_z = 17

xf = np.linspace(-R, R, n_flow_xy)
yf = np.linspace(-R, R, n_flow_xy)
zf = np.linspace(0.0, H, n_flow_z)

Xf, Yf, Zf = np.meshgrid(xf, yf, zf, indexing="ij")
Rf = np.sqrt(Xf**2 + Yf**2)
inside_flow = Rf <= R

# PyVista StructuredGrid uses VTK / Fortran ordering.
RawFlowX, RawFlowY, RawFlowZ = evaluate_mode_set_cartesian(
    raw_modes, R, H, Xf, Yf, Zf, order="F"
)

flow_grid = pv.StructuredGrid(Xf, Yf, Zf)

# Seed points at several radii/heights for a CFD-style 3D picture.
seed_points = []
for rho in (0.32 * R, 0.68 * R):
    for zz in np.linspace(0.15 * H, 0.85 * H, 4):
        for th in np.linspace(0.0, 2 * np.pi, 9, endpoint=False):
            seed_points.append(
                [rho * np.cos(th), rho * np.sin(th), zz]
            )

seed_source = pv.PolyData(np.asarray(seed_points))

# Transparent cylinder for domain context.
cylinder_surface = pv.Cylinder(
    center=(0.0, 0.0, H / 2),
    direction=(0.0, 0.0, 1.0),
    radius=R,
    height=H,
    resolution=128,
    capping=True,
)

# ---------- cross-section ----------
n_cross = 121

xc = np.linspace(-R, R, n_cross)
yc = np.linspace(-R, R, n_cross)

Xc, Yc = np.meshgrid(xc, yc, indexing="xy")
Rc = np.sqrt(Xc**2 + Yc**2)
inside_cross = Rc <= R

# Avoid a highly symmetric axial plane.
z_slice = 0.37 * H

Xci = Xc[inside_cross]
Yci = Yc[inside_cross]
Zci = np.full_like(Xci, z_slice)

RawCrossX, RawCrossY, RawCrossZ = evaluate_mode_set_cartesian(
    raw_modes, R, H, Xci, Yci, Zci
)

# Animation sample times.
animation_index = np.unique(
    np.round(
        np.linspace(0, n_time - 1, n_animation_frames)
    ).astype(int)
)

# ================================================================
# 11. HELPERS FOR ANIMATION FRAMES
# ================================================================

def mpl_figure_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def current_raw_coeff(k):
    # phi1_coeff is expressed in the orthonormal linear basis.
    return Vlin @ phi1_coeff[:, k]


# ================================================================
# 12. FIGURE 1 GIF:
#     instantaneous eigenvalue curves lambda_j(t)
# ================================================================
frames1 = []

curve_colors = plt.cm.tab10(np.linspace(0, 1, n_eigen_curves))

for k in animation_index:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=110)

    for j in range(n_eigen_curves):
        ax.plot(
            t_solution,
            lambda_frozen[j],
            lw=1.8,
            color=curve_colors[j],
            label=fr"$\lambda_{j+1}(t)$",
        )
        ax.scatter(
            [t_solution[k]],
            [lambda_frozen[j, k]],
            s=35,
            color=curve_colors[j],
            zorder=5,
        )

    ax.axvline(t_solution[k], ls="--", lw=1.0, color="black")
    ax.set_xlabel("physical time $t$")
    ax.set_ylabel(r"frozen eigenvalue $\lambda_j(t)$")
    ax.set_title(
        "Instantaneous frozen spectrum\n"
        + fr"$t={t_solution[k]:.3f}$, "
        + fr"$\lambda_1={lambda_frozen[0,k]:.5f}$"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    frames1.append(mpl_figure_to_rgb(fig))
    plt.close(fig)

imageio.mimsave(
    "/content/figure1_frozen_spectrum.gif",
    frames1,
    fps=gif_fps,
)

# ================================================================
# 13. FIGURE 2 GIF:
#     TRUE 3D HYDRODYNAMICS-STYLE MAXWELL EIGENFIELD
#
# This is the main change from the MATLAB version:
#   * mixed toroidal + poloidal basis -> phi_z can be nonzero
#   * PyVista stream tubes, not just line segments/arrows
#   * tube color represents normalized |phi_1|
# ================================================================
frames2 = []

inside_flow_flat = inside_flow.ravel(order="F")

for frame_no, k in enumerate(animation_index, start=1):
    coeff_raw = current_raw_coeff(k)

    ux, uy, uz = combine_modes(
        RawFlowX, RawFlowY, RawFlowZ, coeff_raw
    )

    vectors = np.c_[ux, uy, uz]
    speed = np.linalg.norm(vectors, axis=1)

    max_speed = np.max(speed[inside_flow_flat])
    if max_speed > 0:
        vectors = vectors / max_speed
        speed = speed / max_speed

    # Stop streamlines outside the physical cylinder.
    vectors[~inside_flow_flat] = 0.0
    speed[~inside_flow_flat] = 0.0

    flow_grid["phi1"] = vectors
    flow_grid["speed"] = speed
    flow_grid.set_active_vectors("phi1")

    stream = flow_grid.streamlines_from_source(
        seed_source,
        vectors="phi1",
        integration_direction="both",
        initial_step_length=0.04,
        max_step_length=0.09,
        max_steps=700,
        terminal_speed=3e-4,
        max_time=4.5,
    )

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(760, 580),
    )
    plotter.set_background("white")

    # Transparent cavity.
    plotter.add_mesh(
        cylinder_surface,
        color="lightgray",
        opacity=0.08,
        style="surface",
    )
    plotter.add_mesh(
        cylinder_surface,
        color="gray",
        opacity=0.20,
        style="wireframe",
        line_width=1.0,
    )

    if stream.n_points > 1:
        tubes = stream.tube(radius=0.012)
        plotter.add_mesh(
            tubes,
            scalars="speed",
            cmap="turbo",
            clim=(0.0, 1.0),
            smooth_shading=True,
            scalar_bar_args={
                "title": "|phi_1| / max",
                "vertical": True,
            },
        )
    else:
        plotter.add_points(
            seed_source,
            color="black",
            point_size=4,
        )

    plotter.add_text(
        (
            "Smallest-positive instantaneous Maxwell eigenmode\n"
            f"t = {t_solution[k]:.3f}, "
            f"lambda_1 = {lambda_frozen[0,k]:.5f}"
        ),
        position="upper_left",
        font_size=12,
        color="black",
    )

    plotter.show_bounds(
        bounds=(-R, R, -R, R, 0, H),
        xlabel="x",
        ylabel="y",
        zlabel="z",
        color="black",
        font_size=10,
    )

    plotter.camera_position = [
        (3.2 * R, 3.2 * R, 1.35 * H),
        (0.0, 0.0, 0.50 * H),
        (0.0, 0.0, 1.0),
    ]

    image = plotter.screenshot(return_img=True)
    frames2.append(image)
    plotter.close()

imageio.mimsave(
    "/content/figure2_3d_hydrodynamic_eigenmode.gif",
    frames2,
    fps=gif_fps,
)

# ================================================================
# 14. FIGURE 3 GIF:
#     cross-sectional heat map + transverse streamlines
# ================================================================
frames3 = []

for k in animation_index:
    coeff_raw = current_raw_coeff(k)

    ux_i, uy_i, uz_i = combine_modes(
        RawCrossX, RawCrossY, RawCrossZ, coeff_raw
    )

    mag_i = np.sqrt(ux_i**2 + uy_i**2 + uz_i**2)
    max_mag = np.max(mag_i)
    if max_mag > 0:
        mag_i = mag_i / max_mag

    heat = np.full_like(Xc, np.nan, dtype=float)
    ux_grid = np.full_like(Xc, np.nan, dtype=float)
    uy_grid = np.full_like(Xc, np.nan, dtype=float)

    heat[inside_cross] = mag_i
    ux_grid[inside_cross] = ux_i
    uy_grid[inside_cross] = uy_i

    transverse_speed = np.sqrt(ux_grid**2 + uy_grid**2)
    max_trans = np.nanmax(transverse_speed)

    if max_trans > 0:
        ux_plot = ux_grid / max_trans
        uy_plot = uy_grid / max_trans
    else:
        ux_plot = ux_grid
        uy_plot = uy_grid

    ux_masked = np.ma.array(ux_plot, mask=~inside_cross)
    uy_masked = np.ma.array(uy_plot, mask=~inside_cross)
    heat_masked = np.ma.array(heat, mask=~inside_cross)

    fig, ax = plt.subplots(figsize=(6.3, 5.6), dpi=110)

    im = ax.pcolormesh(
        Xc,
        Yc,
        heat_masked,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    # Cross-sectional flow topology.
    ax.streamplot(
        xc,
        yc,
        ux_masked,
        uy_masked,
        density=1.35,
        color="black",
        linewidth=0.55,
        arrowsize=0.7,
    )

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            R,
            fill=False,
            color="black",
            lw=1.2,
        )
    )

    ax.set_aspect("equal")
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        (
            fr"$|\phi_1|$ and transverse field lines at "
            fr"$z={z_slice:.3f}$"
            "\n"
            fr"$t={t_solution[k]:.3f}$, "
            fr"$\lambda_1={lambda_frozen[0,k]:.5f}$"
        )
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$|\phi_1|/\max|\phi_1|$")

    fig.tight_layout()
    frames3.append(mpl_figure_to_rgb(fig))
    plt.close(fig)

imageio.mimsave(
    "/content/figure3_cross_section.gif",
    frames3,
    fps=gif_fps,
)

print("\nSaved:")
print("/content/figure1_frozen_spectrum.gif")
print("/content/figure2_3d_hydrodynamic_eigenmode.gif")
print("/content/figure3_cross_section.gif")

# ================================================================
# 15. DISPLAY ANIMATIONS INLINE IN COLAB
# ================================================================
display(Image(filename="/content/figure1_frozen_spectrum.gif"))
display(Image(filename="/content/figure2_3d_hydrodynamic_eigenmode.gif"))
display(Image(filename="/content/figure3_cross_section.gif"))


# ================================================================
# 16. CONVERT GIF OUTPUTS TO MP4 FOR GITHUB
# ================================================================
import subprocess
from IPython.display import Video


def gif_to_mp4(gif_file, mp4_file, fps=4):
    cmd = [
        "ffmpeg", "-y",
        "-i", gif_file,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-movflags", "+faststart",
        mp4_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Saved:", mp4_file)


video1_file = "/content/figure1_frozen_spectrum.mp4"
video2_file = "/content/figure2_3d_hydrodynamic_eigenmode.mp4"
video3_file = "/content/figure3_cross_section.mp4"

gif_to_mp4("/content/figure1_frozen_spectrum.gif", video1_file, gif_fps)
gif_to_mp4("/content/figure2_3d_hydrodynamic_eigenmode.gif", video2_file, gif_fps)
gif_to_mp4("/content/figure3_cross_section.gif", video3_file, gif_fps)

print("\nMP4 outputs:")
print(video1_file)
print(video2_file)
print(video3_file)
