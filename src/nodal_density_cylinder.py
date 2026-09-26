import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.special import jv, jn_zeros
from scipy.ndimage import gaussian_filter
import imageio.v2 as imageio
from IPython.display import Video, display

# ================================================================
# 1. PARAMETERS
# ================================================================

rng = np.random.default_rng(1)

R = 1.0
H = 2.0

m_max = 5
ell_max = 5
p_max = 6

n_modes_to_animate = 40

# Approximate nodal set:
#     |phi| <= nodal_tol * max|phi|
nodal_tol = 0.035

# Gaussian smoothing in grid-point units
sigma = 1.0

# Display log(1 + alpha_heat * density)
alpha_heat = 80.0

exclude_boundary_nodes = True

# MP4 playback speed.
# The MATLAB frameDelay=0.8 corresponds to about 1.25 fps.
# fps=2 gives a ~20 s video for 40 modes.
fps = 2

# Balanced Colab resolution.
# For closer MATLAB resolution, change to:
#     n_xy = 35; n_z = 35; n_cross = 201
n_xy = 31
n_z = 31
n_cross = 181

# Plotting only: cumulative density is still computed on the full grid,
# but Figure 2 draws at most this many points for faster video rendering.
max_density_plot_points = 9000
max_current_plot_points = 7000

# The MATLAB file uses H/2.  At H/2 every even-p mode vanishes on the
# entire cross-section.  0.37*H avoids that systematic artifact.
# Set z_slice = H/2 for an exact translation of the MATLAB choice.
z_slice = 0.37 * H

# Output MP4 files
video1_file = "/content/figure1_current_nodal_set.mp4"
video2_file = "/content/figure2_cumulative_nodal_density_3d.mp4"
video3_file = "/content/figure3_cross_section_nodal_density.mp4"


# ================================================================
# 2. CYLINDRICAL DIRICHLET EIGENFUNCTION
#
# phi_{m,ell,p}(r,theta,z)
#   = J_m(alpha_{m,ell} r/R)
#     cos(m theta)
#     sin(p pi z/H)
# ================================================================

def evaluate_cylinder_mode(m, p, alpha, R, H, r, theta, z):
    return (
        jv(m, alpha * r / R)
        * np.cos(m * theta)
        * np.sin(p * np.pi * z / H)
    )


# ================================================================
# 3. BUILD AND SORT THE MODE LIST
# ================================================================

modes = []

for m in range(m_max + 1):
    zeros = jn_zeros(m, ell_max)

    for ell in range(1, ell_max + 1):
        alpha = float(zeros[ell - 1])

        for p in range(1, p_max + 1):
            lam = (alpha / R) ** 2 + (p * np.pi / H) ** 2

            modes.append({
                "m": m,
                "ell": ell,
                "p": p,
                "alpha": alpha,
                "lambda": lam,
            })

modes.sort(key=lambda item: item["lambda"])
modes = modes[: min(n_modes_to_animate, len(modes))]
n_modes_to_animate = len(modes)

print(f"Recording first {n_modes_to_animate} cylindrical eigenmodes.")
print("First few modes:")
for j, mode in enumerate(modes[:8], start=1):
    print(
        f"  j={j:2d}: "
        f"(m,ell,p)=({mode['m']},{mode['ell']},{mode['p']}), "
        f"lambda={mode['lambda']:.6f}"
    )


# ================================================================
# 4. 3D CYLINDER GRID
# ================================================================

x = np.linspace(-R, R, n_xy)
y = np.linspace(-R, R, n_xy)
z = np.linspace(0.0, H, n_z)

# indexing="xy" matches MATLAB meshgrid ordering conceptually.
X, Y, Z = np.meshgrid(x, y, z, indexing="xy")

RR = np.sqrt(X**2 + Y**2)
TH = np.arctan2(Y, X)

inside = RR <= R + 1e-12

dx = 2 * R / (n_xy - 1)
dz = H / (n_z - 1)

boundary_margin_r = 1.5 * dx
boundary_margin_z = 1.5 * dz

interior = (
    (RR < R - boundary_margin_r)
    & (Z > boundary_margin_z)
    & (Z < H - boundary_margin_z)
)

# Fixed whole-cylinder point set for Figure 2
inside_flat_indices = np.flatnonzero(inside.ravel())

if inside_flat_indices.size > max_density_plot_points:
    density_plot_flat_indices = rng.choice(
        inside_flat_indices,
        size=max_density_plot_points,
        replace=False,
    )
else:
    density_plot_flat_indices = inside_flat_indices

Xflat = X.ravel()
Yflat = Y.ravel()
Zflat = Z.ravel()

x_density_plot = Xflat[density_plot_flat_indices]
y_density_plot = Yflat[density_plot_flat_indices]
z_density_plot = Zflat[density_plot_flat_indices]


# ================================================================
# 5. HIGH-RESOLUTION CROSS-SECTION GRID
# ================================================================

xc = np.linspace(-R, R, n_cross)
yc = np.linspace(-R, R, n_cross)

Xc, Yc = np.meshgrid(xc, yc, indexing="xy")

Rc = np.sqrt(Xc**2 + Yc**2)
THc = np.arctan2(Yc, Xc)

inside_cross = Rc <= R + 1e-12
Zc = np.full_like(Rc, z_slice)


# ================================================================
# 6. CUMULATIVE NODAL-DENSITY STORAGE
# ================================================================

density_accum_3d = np.zeros_like(X, dtype=float)
density_accum_cross = np.zeros_like(Xc, dtype=float)


# ================================================================
# 7. VIDEO HELPERS
# ================================================================

def figure_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def make_writer(filename, fps):
    # Figures below are 640 x 560 pixels, both divisible by 16,
    # so H.264/yuv420p encoding is safe.
    return imageio.get_writer(
        filename,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        ffmpeg_params=["-movflags", "+faststart"],
    )


# ================================================================
# 8. CREATE THE THREE FIGURES ONCE
# ================================================================

# ------------------------------------------------
# Figure 1: current interior nodal set
# ------------------------------------------------

fig1 = plt.figure(figsize=(6.4, 5.6), dpi=100)
ax1 = fig1.add_subplot(111, projection="3d")

sc1 = ax1.scatter(
    [],
    [],
    [],
    s=8,
    alpha=0.75,
)

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

ax1.set_xlim(-R, R)
ax1.set_ylim(-R, R)
ax1.set_zlim(0.0, H)

ax1.set_box_aspect((2 * R, 2 * R, H))
ax1.view_init(elev=25, azim=35)
ax1.grid(True)


# ------------------------------------------------
# Figure 2: cumulative nodal density in 3D
# ------------------------------------------------

fig2 = plt.figure(figsize=(6.4, 5.6), dpi=100)
ax2 = fig2.add_subplot(111, projection="3d")

initial_colors = np.zeros(x_density_plot.size)
initial_sizes = np.full(x_density_plot.size, 4.0)

sc2 = ax2.scatter(
    x_density_plot,
    y_density_plot,
    z_density_plot,
    c=initial_colors,
    s=initial_sizes,
    cmap="viridis",
    vmin=0.0,
    vmax=np.log1p(alpha_heat),
    alpha=0.28,
    linewidths=0,
)

cb2 = fig2.colorbar(sc2, ax=ax2, shrink=0.72, pad=0.08)
cb2.set_label(r"$\log(1+\alpha D_J)$")

ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

ax2.set_xlim(-R, R)
ax2.set_ylim(-R, R)
ax2.set_zlim(0.0, H)

ax2.set_box_aspect((2 * R, 2 * R, H))
ax2.view_init(elev=25, azim=35)
ax2.grid(True)


# ------------------------------------------------
# Figure 3: cumulative density on one cross-section
# ------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(6.4, 5.6), dpi=100)

initial_cross = np.full_like(Xc, np.nan, dtype=float)

im3 = ax3.imshow(
    initial_cross,
    extent=(-R, R, -R, R),
    origin="lower",
    cmap="viridis",
    vmin=0.0,
    vmax=np.log1p(alpha_heat),
    interpolation="nearest",
)

ax3.add_patch(
    Circle(
        (0.0, 0.0),
        R,
        fill=False,
        color="black",
        lw=1.0,
    )
)

cb3 = fig3.colorbar(im3, ax=ax3)
cb3.set_label(r"$\log(1+\alpha D_J)$")

ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_aspect("equal")
ax3.set_xlim(-R, R)
ax3.set_ylim(-R, R)


# ================================================================
# 9. RECORD ALL THREE MP4s IN ONE PASS THROUGH THE MODES
# ================================================================

writer1 = make_writer(video1_file, fps)
writer2 = make_writer(video2_file, fps)
writer3 = make_writer(video3_file, fps)

print("\nStarting MP4 recording...\n")

try:
    for j, mode in enumerate(modes, start=1):
        m = mode["m"]
        ell = mode["ell"]
        p = mode["p"]
        alpha = mode["alpha"]
        lam = mode["lambda"]

        # ========================================================
        # Current 3D eigenfunction
        # ========================================================

        Phi = evaluate_cylinder_mode(
            m, p, alpha, R, H, RR, TH, Z
        )

        max_phi = np.max(np.abs(Phi[inside]))

        if max_phi > 0:
            if exclude_boundary_nodes:
                current_mask = (
                    interior
                    & (np.abs(Phi) <= nodal_tol * max_phi)
                )
            else:
                current_mask = (
                    inside
                    & (np.abs(Phi) <= nodal_tol * max_phi)
                )
        else:
            current_mask = np.zeros_like(Phi, dtype=bool)

        # ========================================================
        # Cumulative 3D nodal density
        # ========================================================

        density_accum_3d += current_mask.astype(float)

        density_average_3d = density_accum_3d / j

        # This replaces MATLAB convn(..., G3, 'same').
        density_smooth_3d = gaussian_filter(
            density_average_3d,
            sigma=sigma,
            mode="constant",
            cval=0.0,
            truncate=5.0,
        )

        # ========================================================
        # Current cross-sectional eigenfunction
        # ========================================================

        Phi_cross = evaluate_cylinder_mode(
            m, p, alpha, R, H, Rc, THc, Zc
        )

        max_cross = np.max(np.abs(Phi_cross[inside_cross]))

        if max_cross > 0:
            cross_current_mask = (
                inside_cross
                & (np.abs(Phi_cross) <= nodal_tol * max_cross)
            )
        else:
            cross_current_mask = np.zeros_like(
                Phi_cross,
                dtype=bool,
            )

        # ========================================================
        # Cumulative cross-sectional nodal density
        # ========================================================

        density_accum_cross += cross_current_mask.astype(float)

        density_average_cross = density_accum_cross / j

        # This replaces MATLAB conv2(..., G2, 'same').
        density_smooth_cross = gaussian_filter(
            density_average_cross,
            sigma=sigma,
            mode="constant",
            cval=0.0,
            truncate=5.0,
        )

        # ========================================================
        # FIGURE 1: current interior nodal set
        # ========================================================

        current_flat = np.flatnonzero(current_mask.ravel())

        if current_flat.size > max_current_plot_points:
            current_flat = rng.choice(
                current_flat,
                size=max_current_plot_points,
                replace=False,
            )

        x_cur = Xflat[current_flat]
        y_cur = Yflat[current_flat]
        z_cur = Zflat[current_flat]

        sc1._offsets3d = (
            x_cur,
            y_cur,
            z_cur,
        )

        ax1.set_title(
            "Current interior nodal set in cylinder\n"
            + rf"$j={j}$, "
            + rf"$(m,\ell,p)=({m},{ell},{p})$, "
            + rf"$\lambda_j={lam:.4f}$"
        )

        fig1.tight_layout()

        writer1.append_data(
            figure_to_rgb(fig1)
        )

        # ========================================================
        # FIGURE 2: cumulative nodal density in the cylinder
        # ========================================================

        density_flat = density_smooth_3d.ravel()
        density_plot = density_flat[density_plot_flat_indices]

        heat_plot = np.log1p(
            alpha_heat * density_plot
        )

        dmax = np.max(density_plot)

        if dmax > 0:
            size_data = (
                3.0
                + 28.0 * density_plot / dmax
            )
        else:
            size_data = np.full_like(
                density_plot,
                3.0,
            )

        sc2.set_array(heat_plot)
        sc2.set_sizes(size_data)

        ax2.set_title(
            "Cumulative nodal density in cylinder\n"
            + rf"first $J={j}$ modes, "
            + rf"current $\lambda_j={lam:.4f}$"
        )

        fig2.tight_layout()

        writer2.append_data(
            figure_to_rgb(fig2)
        )

        # ========================================================
        # FIGURE 3: cross-sectional cumulative density
        # ========================================================

        cross_heat = np.log1p(
            alpha_heat * density_smooth_cross
        )

        cross_heat = cross_heat.copy()
        cross_heat[~inside_cross] = np.nan

        im3.set_data(
            cross_heat
        )

        ax3.set_title(
            "Cross-sectional cumulative nodal density\n"
            + rf"$z={z_slice:.3f}$, "
            + rf"$J={j}$, "
            + rf"$(m,\ell,p)=({m},{ell},{p})$, "
            + rf"$\lambda_j={lam:.4f}$"
        )

        fig3.tight_layout()

        writer3.append_data(
            figure_to_rgb(fig3)
        )

        print(
            f"mode {j:2d}/{n_modes_to_animate}: "
            f"(m,ell,p)=({m},{ell},{p}), "
            f"lambda={lam:.6f}"
        )

finally:
    writer1.close()
    writer2.close()
    writer3.close()

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


# ================================================================
# 10. RESULTS
# ================================================================

print("\nSaved MP4 videos:")
print(video1_file)
print(video2_file)
print(video3_file)

display(Video(video1_file, embed=True))
display(Video(video2_file, embed=True))
display(Video(video3_file, embed=True))
