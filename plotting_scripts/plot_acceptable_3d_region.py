# Visualize the acceptable S/N region as an isosurface at s2n_min.

import sys
import zlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import image as mpimg
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import plotly.graph_objects as go
import ipdb

# Maximally distinct categorical palette (spaced hues; sequential assignment).
_DETECTOR_COLORS = [
    mcolors.to_rgb(c)
    for c in (
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#00a9e0",  # cyan
        "#1b9e77",  # teal
        "#e6ab02",  # gold
        "#7570b3",  # indigo
        "#66a61e",  # olive
        "#d95f02",  # dark orange
        "#e7298a",  # deep pink
        "#666666",  # gray
        "#a6d854",  # lime
        "#8da0cb",  # periwinkle
        "#b15928",  # rust
        "#6a3d9a",  # deep purple
        "#33a02c",  # strong green
        "#ff1493",  # fuchsia
        "#008080",  # deep teal
        "#b2df8a",  # pale green
        "#cab2d6",  # lavender
        "#ffff33",  # yellow
        "#fb9a99",  # light red
        "#1f78b4",  # medium blue
        "#b8860b",  # dark goldenrod
        "#00ced1",  # turquoise
        "#8b0000",  # dark red
        "#7cfc00",  # chartreuse
        "#4169e1",  # royal blue
    )
]


def _color_for_index(i):
    """Return a high-contrast RGB color unique for consecutive indices."""
    if i < len(_DETECTOR_COLORS):
        return _DETECTOR_COLORS[i]
    # Golden-angle hue spacing for overflow beyond the fixed palette.
    hue = (i * 0.618033988749895) % 1.0
    sat = 0.85 if (i % 2 == 0) else 0.65
    val = 0.9 if (i % 3 != 2) else 0.55
    return tuple(mcolors.hsv_to_rgb((hue, sat, val)))


def _color_from_name(name):
    """Fallback name-hashed color (prefer ``_color_for_index`` for uniqueness)."""
    idx = zlib.adler32(name.encode("utf-8")) % len(_DETECTOR_COLORS)
    return _DETECTOR_COLORS[idx]


# Region-of-interest box: (wavelength, QE, dark current) in physical units.
class DetectorROI:
    def __init__(self, name, wavelength_0, wavelength_1, qe_0, qe_1, dc_0, dc_1, color=None):
        self.name = name
        self.wavelength_0 = wavelength_0
        self.wavelength_1 = wavelength_1
        self.qe_0 = qe_0
        self.qe_1 = qe_1
        self.dc_0 = dc_0
        self.dc_1 = dc_1
        self.color = color if color is not None else _color_from_name(name)

    def plot(self, ax, mode="3d", outline=True):
        if mode == "3d":
            _plot_bounding_box(
                ax,
                self.wavelength_0,
                self.wavelength_1,
                self.qe_0,
                self.qe_1,
                self.dc_0,
                self.dc_1,
                color=self.color,
                label=self.name,
            )
        elif mode == "wavel_dc":
            _plot_filled_rectangle(
                ax,
                self.wavelength_0,
                self.wavelength_1,
                self.dc_0,
                self.dc_1,
                color=self.color,
                label=self.name,
                outline=outline,
            )
        elif mode == "wavel_qe":
            _plot_filled_rectangle(
                ax,
                self.wavelength_0,
                self.wavelength_1,
                self.qe_0,
                self.qe_1,
                color=self.color,
                label=self.name,
                outline=outline,
            )

def _plot_bounding_box(ax, x0, x1, y0, y1, z0, z1, **plot_kwargs):
    """Filled box with plot axes x=wavelength, y=QE, z=dark current."""
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    face_idx = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
    )
    color = plot_kwargs.pop("color", plot_kwargs.pop("facecolor", "red"))
    alpha = plot_kwargs.pop("alpha", 0.4)
    label = plot_kwargs.pop("label", None)
    box = Poly3DCollection(
        [corners[list(idx)] for idx in face_idx],
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        label=label,
        **plot_kwargs,
    )
    ax.add_collection3d(box)


def _plot_filled_rectangle(ax, x0, x1, y0, y1, color="red", alpha=0.4, label=None, outline=True):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=color,
            edgecolor="black" if outline else "none",
            linewidth=0.8 if outline else 0.0,
            alpha=alpha,
            label=label,
        )
    )


def _detector_label_anchors(detector, mode):
    """Return (x_left, x_right, y_top, y_bottom) for the ROI line in data coordinates."""
    x_left = float(detector.wavelength_0)
    x_right = float(detector.wavelength_1)
    if mode == "wavel_dc":
        y_top = float(detector.dc_1)
        y_bottom = float(detector.dc_0)
    elif mode == "wavel_qe":
        y_top = float(detector.qe_1)
        y_bottom = float(detector.qe_0)
    else:
        raise ValueError(f"Unsupported label mode: {mode!r}")
    return x_left, x_right, y_top, y_bottom


def _visible_roi_segment(ax, x_left, x_right, y_top, y_bottom):
    """
    Clip an ROI to the axes data window.

    Returns (vis_x0, vis_x1, vis_y) for labeling, or None if the ROI does not
    intersect the plotted area. ``vis_y`` is the top of the visible segment
    (clamped into the axes), so bottom-edge stubs still get an in-plot anchor.
    """
    x_lo, x_hi = sorted(ax.get_xlim())
    y_lo, y_hi = sorted(ax.get_ylim())
    x0 = min(x_left, x_right)
    x1 = max(x_left, x_right)
    y0 = min(y_top, y_bottom)
    y1 = max(y_top, y_bottom)

    vis_x0 = max(x0, x_lo)
    vis_x1 = min(x1, x_hi)
    vis_y0 = max(y0, y_lo)
    vis_y1 = min(y1, y_hi)
    # Allow zero-height intersections (thin ROI sitting on the axis edge).
    if vis_x0 >= vis_x1 or vis_y0 > vis_y1:
        return None
    return vis_x0, vis_x1, vis_y1


def _bbox_inside(inner, outer, pad_px=2.0):
    """True if ``inner`` lies fully inside ``outer`` (with padding)."""
    return (
        inner.x0 >= outer.x0 + pad_px
        and inner.x1 <= outer.x1 - pad_px
        and inner.y0 >= outer.y0 + pad_px
        and inner.y1 <= outer.y1 - pad_px
    )


def _bboxes_overlap(b1, b2, pad_px=12.0):
    """True if two display-coordinate Bbox objects overlap (with padding)."""
    return not (
        b1.x1 + pad_px < b2.x0
        or b2.x1 + pad_px < b1.x0
        or b1.y1 + pad_px < b2.y0
        or b2.y1 + pad_px < b1.y0
    )


# Just above the line.
_LABEL_OFFSET_ABOVE = (0.0, 1.5)


def _label_display_bbox(
    x_disp, y_disp, text_w, text_h, dx, dy, px_per_pt, va="bottom", ha="left", rotation=0.0
):
    """Label bounding box in display pixels for a given offset (AABB if rotated)."""
    from matplotlib.transforms import Bbox

    px = x_disp + dx * px_per_pt
    py = y_disp + dy * px_per_pt
    # Local corners relative to the text alignment / rotation pivot.
    if ha == "left":
        xs = (0.0, text_w)
    elif ha == "right":
        xs = (-text_w, 0.0)
    else:
        xs = (-0.5 * text_w, 0.5 * text_w)
    if va == "bottom":
        ys = (0.0, text_h)
    elif va == "top":
        ys = (-text_h, 0.0)
    else:
        ys = (-0.5 * text_h, 0.5 * text_h)

    corners = [(x, y) for x in xs for y in ys]
    if rotation:
        theta = np.deg2rad(rotation)
        c, s = np.cos(theta), np.sin(theta)
        corners = [(x * c - y * s, x * s + y * c) for x, y in corners]

    x0 = px + min(x for x, _ in corners)
    x1 = px + max(x for x, _ in corners)
    y0 = py + min(y for _, y in corners)
    y1 = py + max(y for _, y in corners)
    return Bbox.from_extents(x0, y0, x1, y1)


def _estimate_text_size_px(fig, name, fontsize, px_per_pt, fontweight="bold"):
    """Measure annotation size in display pixels, with a small safety inflate."""
    renderer = fig.canvas.get_renderer()
    tmp = fig.text(
        0, 0, name, fontsize=fontsize, fontweight=fontweight, visible=False
    )
    try:
        bbox = tmp.get_window_extent(renderer=renderer)
        # Inflate slightly so bold ink / antialiasing does not slip past overlap checks.
        return float(bbox.width) * 1.15, float(bbox.height) * 1.25
    finally:
        tmp.remove()


def _detector_visible_on_ax(ax, detector, mode):
    """True if the detector ROI intersects the axes data window."""
    x_left, x_right, y_top, y_bottom = _detector_label_anchors(detector, mode)
    return _visible_roi_segment(ax, x_left, x_right, y_top, y_bottom) is not None


def _detectors_visible_on_both_panels(ax_dc, ax_qe, detectors):
    """Names of detectors whose ROIs appear in both the DC and QE panels."""
    return {
        detector.name
        for detector in detectors
        if _detector_visible_on_ax(ax_dc, detector, "wavel_dc")
        and _detector_visible_on_ax(ax_qe, detector, "wavel_qe")
    }


def _place_detector_labels(
    ax, detectors, mode, fontsize=8, omit_top_stack=True, bold_names=None
):
    """
    Place detector name annotations just above the ROI lines.

    Default: left-aligned at the visible left end of the ROI (or right-aligned
    at the visible right end if left would leave the axes). Anchors are clipped
    to the plotted window so partially visible bottom/left stubs can still be
    labeled in-place. If a label would overhang the left axes edge, it is
    shifted so its text starts at that edge. Labels that would be out of bounds
    or overlap another on-line label are collected as a top stack; by default
    (``omit_top_stack=True``) those boxed top labels are not drawn.

    ``bold_names``: if given, only those detector names use bold type; others
    are drawn with normal weight (typically detectors visible in both panels).
    """
    if bold_names is None:
        bold_names = {d.name for d in detectors}

    items = []
    for detector in detectors:
        x_left, x_right, y_top, y_bottom = _detector_label_anchors(detector, mode)
        items.append((x_left, x_right, y_top, y_bottom, detector.name, detector.color))
    items.sort(key=lambda t: (t[0], t[2], t[4]))

    fig = ax.figure
    fig.canvas.draw()
    axes_bbox = ax.bbox.frozen()
    px_per_pt = fig.dpi / 72.0
    dx0, dy0 = _LABEL_OFFSET_ABOVE
    va = "bottom"
    left_pad_px = 4.0
    qe_rotate_thresh = 0.65
    qe_rotation_deg = 20.0  # CCW

    on_line = []
    top_stack = []
    placed_bboxes = []

    def _label_rotation(y):
        if mode == "wavel_qe" and y > qe_rotate_thresh:
            return qe_rotation_deg
        return 0.0

    def _evaluate(x, y, ha, text_w, text_h):
        x_disp, y_disp = ax.transData.transform((x, y))
        dx, dy = dx0, dy0
        rotation = _label_rotation(y)
        bbox = _label_display_bbox(
            x_disp,
            y_disp,
            text_w,
            text_h,
            dx,
            dy,
            px_per_pt,
            va=va,
            ha=ha,
            rotation=rotation,
        )
        # Keep labels from being clipped on the left: shift so text starts
        # at the left-hand side of the plotting area.
        if bbox.x0 < axes_bbox.x0 + left_pad_px:
            dx += ((axes_bbox.x0 + left_pad_px) - bbox.x0) / px_per_pt
            bbox = _label_display_bbox(
                x_disp,
                y_disp,
                text_w,
                text_h,
                dx,
                dy,
                px_per_pt,
                va=va,
                ha=ha,
                rotation=rotation,
            )
        return {
            "in_bounds": _bbox_inside(bbox, axes_bbox),
            "overlaps": any(_bboxes_overlap(bbox, other) for other in placed_bboxes),
            "dx": dx,
            "dy": dy,
            "va": va,
            "ha": ha,
            "bbox": bbox,
            "x": x,
            "y": y,
            "rotation": rotation,
        }

    for x_left, x_right, y_top, y_bottom, name, color in items:
        visible = _visible_roi_segment(ax, x_left, x_right, y_top, y_bottom)
        if visible is None:
            top_stack.append((name, color))
            continue

        vis_x0, vis_x1, vis_y = visible
        fontweight = "bold" if name in bold_names else "normal"
        text_w, text_h = _estimate_text_size_px(
            fig, name, fontsize, px_per_pt, fontweight=fontweight
        )

        cand = _evaluate(vis_x0, vis_y, "left", text_w, text_h)
        if (not cand["in_bounds"]) or cand["overlaps"]:
            right = _evaluate(vis_x1, vis_y, "right", text_w, text_h)
            if right["in_bounds"] and not right["overlaps"]:
                cand = right

        if (not cand["in_bounds"]) or cand["overlaps"]:
            top_stack.append((name, color))
            continue

        placed_bboxes.append(cand["bbox"])
        on_line.append(
            (
                cand["x"],
                cand["y"],
                name,
                color,
                cand["dx"],
                cand["dy"],
                cand["va"],
                cand["ha"],
                cand["rotation"],
                fontweight,
            )
        )

    for x, y, name, color, dx_i, dy_i, va_i, ha_i, rotation, fontweight in on_line:
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(dx_i, dy_i),
            textcoords="offset points",
            ha=ha_i,
            va=va_i,
            color=color,
            fontsize=fontsize,
            fontweight=fontweight,
            rotation=rotation,
            rotation_mode="anchor",
            clip_on=True,
            zorder=5,
        )

    # Optionally park overflowing / overlapping labels at the top-left.
    if omit_top_stack:
        return
    line_spacing = 0.045  # axes-fraction step between stacked labels
    for i, (name, color) in enumerate(top_stack):
        fontweight = "bold" if name in bold_names else "normal"
        ax.text(
            0.01,
            0.99 - i * line_spacing,
            name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=color,
            fontsize=fontsize,
            fontweight=fontweight,
            clip_on=True,
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.2,
                "alpha": 0.92,
            },
        )


def _rgb_to_plotly(color, alpha=1.0):
    """Convert matplotlib RGB(A) tuple or named color to a plotly rgba string."""
    if isinstance(color, str):
        r, g, b, a = mcolors.to_rgba(color, alpha=alpha)
    else:
        r, g, b = (float(c) for c in color[:3])
        a = float(alpha) if len(color) < 4 else float(color[3])
        if alpha != 1.0:
            a = float(alpha)
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a:.3f})"


def _detector_box_mesh3d(detector, z_max=None):
    """Plotly Mesh3d for a DetectorROI axis-aligned box."""
    x0, x1 = float(detector.wavelength_0), float(detector.wavelength_1)
    y0, y1 = float(detector.qe_0), float(detector.qe_1)
    z0, z1 = float(detector.dc_0), float(detector.dc_1)
    if z_max is not None:
        z0 = min(z0, z_max)
        z1 = min(z1, z_max)
        if z1 <= z0:
            return None
    # 8 corners, then triangulated faces (two triangles per face).
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    i, j, k = [], [], []
    for a, b, c, d in (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
    ):
        i.extend([a, a])
        j.extend([b, c])
        k.extend([c, d])
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        color=_rgb_to_plotly(detector.color, alpha=0.45),
        flatshading=True,
        name=detector.name,
        showlegend=True,
        hoverinfo="name",
    )


# Per-detector label nudges in 3D data coords (wavelength, QE, dark current).
# Positive QE/DC shifts move the label up along those axes.
_ANNOTATION_OFFSETS_3D = {
    # Lower in QE so it clears a neighboring label.
    "Teledyne": (0.0, -0.5, 0.0),
    # Shift left along wavelength to reduce overlap.
    "Raytheon Si:As": (-9.0, 0.0, 0.0),
}


def _annotation_offset_3d(name):
    """Return (dx, dy, dz) data-coord nudge for a detector label, if any."""
    for key, offset in _ANNOTATION_OFFSETS_3D.items():
        if key.lower() in name.lower():
            return offset
    return (0.0, 0.0, 0.0)


def _detector_label_annotation3d(detector, z_max=None, bold=False):
    """
    Scene annotation at the top-center of a detector ROI box.

    Returns a one-element list (or None). Uses a white backing so colored
    labels stay readable over the blue volume — Plotly scene annotations
    have no true letter stroke.
    """
    x0, x1 = float(detector.wavelength_0), float(detector.wavelength_1)
    y0, y1 = float(detector.qe_0), float(detector.qe_1)
    z0, z1 = float(detector.dc_0), float(detector.dc_1)
    if z_max is not None:
        z0 = min(z0, z_max)
        z1 = min(z1, z_max)
        if z1 <= z0:
            return None
    dx, dy, dz = _annotation_offset_3d(detector.name)
    text = f"<b>{detector.name}</b>" if bold else detector.name
    return [
        dict(
            x=0.5 * (x0 + x1) + dx,
            y=y1 + dy,
            z=z1 + dz,
            text=text,
            showarrow=False,
            font=dict(color=_rgb_to_plotly(detector.color, alpha=1.0), size=11),
            bgcolor="rgba(255,255,255,0.45)",
            bordercolor="rgba(255,255,255,0.45)",
            borderwidth=1,
            borderpad=1,
            xanchor="left",
            yanchor="bottom",
        )
    ]


def _camera_from_zoom_and_angles(
    zoom=1.0,
    angles_deg=(0.0, 0.0, 0.0),
    *,
    base_eye=(2.2, -1.7, 1.4),
    center=(0.0, 0.0, -0.08),
):
    """
    Plotly scene camera from a single zoom and x/y/z rotation vector.

    ``zoom`` scales the camera distance: values > 1 zoom in, values < 1 zoom out.
    ``angles_deg`` is (rx, ry, rz), rotating around wavelength, QE, and dark
    current axes respectively.
    """
    if zoom <= 0:
        raise ValueError(f"Camera zoom must be positive, got {zoom!r}")

    rx, ry, rz = np.deg2rad(np.asarray(angles_deg, dtype=float))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ]
    )
    rot_y = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ]
    )
    rot_z = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    eye = (rot_z @ rot_y @ rot_x @ np.asarray(base_eye, dtype=float)) / float(zoom)
    return dict(
        eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
        center=dict(x=float(center[0]), y=float(center[1]), z=float(center[2])),
    )


def _build_acceptable_volume_figure(
    snr_plot,
    wavel,
    qe,
    dc,
    s2n_min,
    detectors_dict,
    *,
    z_max=0.5,
    wavel_stride=2,
    dc_stride=2,
    annotate_detectors=False,
    show_detector_legend=False,
    title=None,
    bold_fonts=False,
    camera_zoom=1.0,
    camera_angles_deg=(0.0, 0.0, 0.0),
):
    """
    Build a Plotly 3D figure of SNR >= s2n_min with detector ROI boxes.

    Axes: x=wavelength, y=QE, z=dark current.
    """
    dc_plot = np.asarray(dc)
    wavel = np.asarray(wavel)
    qe = np.asarray(qe)
    snr_plot = np.asarray(snr_plot)

    z_keep = dc_plot <= z_max
    if not np.any(z_keep):
        raise ValueError(f"No dark-current samples at or below z_max={z_max}")
    dc_plot = dc_plot[z_keep]
    snr_clip = snr_plot[:, :, z_keep]

    # Extra downsample — kaleido export cost grows quickly with grid size.
    snr_clip = snr_clip[::wavel_stride, :, ::dc_stride]
    wavel = wavel[::wavel_stride]
    dc_plot = dc_plot[::dc_stride]

    X, Y, Z = np.meshgrid(wavel, qe, dc_plot, indexing="ij")
    values = snr_clip.astype(float)
    n_pts = values.size
    print(
        f"Building 3D Plotly figure "
        f"({values.shape[0]}×{values.shape[1]}×{values.shape[2]} = {n_pts} samples)..."
    )

    # Nested isosurfaces approximate a filled volume; much faster than go.Volume.
    fill = go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=float(s2n_min),
        isomax=float(np.nanmax(values)),
        surface_count=6,
        opacity=0.25,
        colorscale="Blues",
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=True),
        name=f"S/N ≥ {s2n_min:g}",
        showlegend=False,
    )
    iso = go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=float(s2n_min),
        isomax=float(s2n_min),
        surface_count=1,
        opacity=0.7,
        colorscale=[[0, "dodgerblue"], [1, "dodgerblue"]],
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name=f"S/N = {s2n_min:g}",
        showlegend=False,
    )

    traces = [fill, iso]
    annotations = []
    for detector in detectors_dict.values():
        mesh = _detector_box_mesh3d(detector, z_max=z_max)
        if mesh is not None:
            mesh.showlegend = bool(show_detector_legend)
            if bold_fonts:
                mesh.name = f"<b>{detector.name}</b>"
            traces.append(mesh)
        if annotate_detectors:
            anns = _detector_label_annotation3d(
                detector, z_max=z_max, bold=bold_fonts
            )
            if anns:
                annotations.extend(anns)

    def _maybe_bold(text):
        return f"<b>{text}</b>" if bold_fonts else text

    axis_label_font = dict(size=14)
    axis_tick_font = dict(size=11)
    title_text = _maybe_bold(title) if title else title
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)) if title else None,
        scene=dict(
            xaxis=dict(
                title=dict(text="Wavelength (um)", font=axis_label_font),
                tickfont=axis_tick_font,
                range=[float(np.min(wavel)), float(np.max(wavel))],
            ),
            yaxis=dict(
                title=dict(text="QE", font=axis_label_font),
                tickfont=axis_tick_font,
                dtick=0.2,
                range=[float(np.min(qe)), 1.0],
            ),
            zaxis=dict(
                title=dict(text="Dark current (e/pix/s)", font=axis_label_font),
                tickfont=axis_tick_font,
                dtick=0.1,
                range=[float(np.min(dc_plot)), float(z_max)],
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.0, z=1.0),
            camera=_camera_from_zoom_and_angles(
                zoom=camera_zoom,
                angles_deg=camera_angles_deg,
            ),
            annotations=annotations,
        ),
        margin=dict(l=80, r=80, t=60, b=90),
        showlegend=bool(show_detector_legend),
        legend=dict(
            itemsizing="constant",
            font=dict(size=10, family="Arial Black" if bold_fonts else None),
        ),
        font=dict(size=10),
    )
    return fig


def _plot_acceptable_volume_plotly(
    snr_plot,
    wavel,
    qe,
    dc,
    s2n_min,
    detectors_dict,
    *,
    z_max=0.5,
    width=3200,
    height=2600,
    # NB: kaleido renders width*scale × height*scale pixels; above roughly
    # 16000 px per side the export silently emits non-PNG data
    # ("SyntaxError: not a PNG file"), so keep the product modest.
    scale=1,
    wavel_stride=2,
    dc_stride=2,
    camera_zoom=1.0,
    camera_angles_deg=(0.0, 0.0, 0.0),
):
    """
    Static Plotly rendering of SNR >= s2n_min for embedding in matplotlib.

    Returns a numpy RGB(A) image array suitable for ``ax.imshow``.
    """
    fig = _build_acceptable_volume_figure(
        snr_plot,
        wavel,
        qe,
        dc,
        s2n_min,
        detectors_dict,
        z_max=z_max,
        wavel_stride=wavel_stride,
        dc_stride=dc_stride,
        annotate_detectors=False,
        show_detector_legend=False,
        title=None,
        camera_zoom=camera_zoom,
        camera_angles_deg=camera_angles_deg,
    )
    # Match matplotlib default axis label / tick sizes used in the 2D panels.
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(font=dict(size=24)), tickfont=dict(size=18)),
            yaxis=dict(title=dict(font=dict(size=24)), tickfont=dict(size=18)),
            zaxis=dict(title=dict(font=dict(size=24)), tickfont=dict(size=18)),
        ),
        showlegend=False,
    )
    print("Exporting Plotly figure to PNG (kaleido)...")
    buf = io.BytesIO()
    fig.write_image(buf, format="png", width=width, height=height, scale=scale)
    buf.seek(0)
    print("3D panel export done.")
    return mpimg.imread(buf, format="png")


def _show_interactive_volume_plotly(
    snr_plot,
    wavel,
    qe,
    dc,
    s2n_min,
    detectors_dict,
    *,
    z_max=0.5,
    wavel_stride=2,
    dc_stride=2,
    output_html=None,
    camera_zoom=1.0,
    camera_angles_deg=(0.0, 0.0, 0.0),
):
    """Save and open a standalone interactive 3D plot (no local server needed)."""
    fig = _build_acceptable_volume_figure(
        snr_plot,
        wavel,
        qe,
        dc,
        s2n_min,
        detectors_dict,
        z_max=z_max,
        wavel_stride=wavel_stride,
        dc_stride=dc_stride,
        annotate_detectors=True,
        show_detector_legend=True,
        title=f"Acceptable S/N region (interactive, S/N ≥ {s2n_min:g})",
        bold_fonts=True,
        camera_zoom=camera_zoom,
        camera_angles_deg=camera_angles_deg,
    )
    if output_html is None:
        output_html = (
            Path.home()
            / "Downloads"
            / "acceptable_s2n_region_3d_interactive.html"
        )
    output_html = Path(output_html).expanduser().resolve()
    print(f"Saving interactive 3D Plotly figure to {output_html}...")
    fig.write_html(
        str(output_html),
        include_plotlyjs=True,
        full_html=True,
        auto_open=True,
    )
    return fig


def _save_annotated_volume_png(
    snr_plot,
    wavel,
    qe,
    dc,
    s2n_min,
    detectors_dict,
    *,
    z_max=0.5,
    wavel_stride=2,
    dc_stride=2,
    output_png=None,
    width=800,
    height=600,
    scale=5,
    camera_zoom=1.0,
    camera_angles_deg=(0.0, 0.0, 0.0),
):
    """Save the annotated 3D volume (with detector labels) as a high-res PNG."""
    fig = _build_acceptable_volume_figure(
        snr_plot,
        wavel,
        qe,
        dc,
        s2n_min,
        detectors_dict,
        z_max=z_max,
        wavel_stride=wavel_stride,
        dc_stride=dc_stride,
        annotate_detectors=True,
        show_detector_legend=True,
        title=f"Acceptable S/N region (S/N ≥ {s2n_min:g})",
        bold_fonts=True,
        camera_zoom=camera_zoom,
        camera_angles_deg=camera_angles_deg,
    )
    if output_png is None:
        output_png = (
            Path.home()
            / "Downloads"
            / "junk_acceptable_s2n_region_3d_annotated.png"
        )
    output_png = Path(output_png).expanduser().resolve()
    print(
        f"Exporting annotated 3D Plotly figure to PNG "
        f"({width}×{height} @ scale={scale})..."
    )
    fig.write_image(
        str(output_png), format="png", width=width, height=height, scale=scale
    )
    print(f"Saved annotated 3D plot to {output_png}")
    return fig


def _plot_roi_rectangle(ax, x0, x1, y0, y1, **rect_kwargs):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            **rect_kwargs,
        )
    )

def _read_csv(filename):

    df = pd.read_csv(filename, delimiter=',')
    detectors_dict = {}

    for i, (_index, row) in enumerate(df.iterrows()):
        # define detectors
        
        offset = 0.02 # sets line thickness in plot
        detect_name = row['plot_string_name']
        detectors_dict[detect_name] = DetectorROI(
                                                    name=detect_name,
                                                    wavelength_0=row['cutoff_blue_um'], 
                                                    wavelength_1=row['cutoff_red_um'], 
                                                    qe_0=row['qe']-offset, 
                                                    qe_1=row['qe'], 
                                                    dc_0=row['dark_current_single_val']-offset, 
                                                    dc_1=row['dark_current_single_val'],
                                                    color=_color_for_index(i),
                                                    )


    return detectors_dict

def _plot_marginalized_panel(ax, x, y, snr_2d, xlabel, ylabel, title):
    # contourf expects Z shape (len(y), len(x)); snr_2d is (len(x), len(y)).
    ax.contourf(
        x,
        y,
        snr_2d.T,
        levels=np.linspace(s2n_min, snr_2d.max(), 20),
        cmap="Blues",
        alpha=0.85,
    )
    ax.contour(x, y, snr_2d.T, levels=[s2n_min], colors="navy", linewidths=1)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, pad=12, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

#################

from load_s2n_cube import load_s2n_cube, print_cube_statistics

# Path to S/N cubes HDF5 written by save_s2n_cube() in calculator.py
s2n_hdf5_path = "/Users/eckhartspalding/Documents/git.repos/life_detectors/plotting_scripts/large_sweep_test/qe_0.80_s2n_cube.hdf5"

detectors_dict = _read_csv(filename='/Users/eckhartspalding/Documents/git.repos/life_detectors/dev_notebooks/data/detector_catalog_truncated_for_paper.csv')

cube = load_s2n_cube(s2n_hdf5_path)
print_cube_statistics(cube)

snr_cube = cube.snr
wavelength = cube.wavelength
dark_current = cube.dark_current
qe_list = cube.qe

s2n_min = 3.0

# Subsample for display only (does not change the saved HDF5 data).
dc_stride = 50   # 0.001 e/pix/s steps -> 0.05 e/pix/s bins
wavel_stride = 5
snr_vis = snr_cube[::wavel_stride, ::dc_stride, :]
wavel_vis = wavelength[::wavel_stride]
dc_vis = dark_current[::dc_stride]

print(
    f"Visualization grid shape (wavelength, DC, QE): {snr_vis.shape} "
    f"(subsampled from {snr_cube.shape})"
)

# Plot axes: x=wavelength, y=QE, z=dark current (transpose from cube order wavel, DC, QE).
snr_plot = snr_vis.transpose(0, 2, 1)

if snr_plot.max() < s2n_min:
    raise ValueError(f"No voxels reach S/N >= {s2n_min:g}; isosurface is empty.")
if snr_plot.min() >= s2n_min:
    raise ValueError(f"All voxels exceed S/N >= {s2n_min:g}; no boundary to extract.")

# Marginalize over QE and DC (max S/N projection onto each plane).
snr_wavel_dc = np.max(snr_plot, axis=1)  # shape (wavelength, DC)
snr_wavel_qe = np.max(snr_plot, axis=2)  # shape (wavelength, QE)

# 3D camera controls for the annotated high-res PNG.
# camera_zoom > 1 zooms in; < 1 zooms out.
# camera_angles_deg = (wavelength-axis, QE-axis, dark-current-axis) in degrees.
camera_zoom = 1.0
camera_angles_deg = (10.0, 20.0, -30.0)

fig = plt.figure(figsize=(14, 5.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)

ax_dc = fig.add_subplot(gs[0, 0])
_plot_marginalized_panel(
    ax_dc,
    wavel_vis,
    dc_vis,
    snr_wavel_dc,
    "Wavelength (um)",
    "Dark current (e/pix/s)",
    f"Marginalized over QE (max S/N, contour at {s2n_min:g})",
)
for detector in detectors_dict.values():
    detector.plot(ax_dc, mode="wavel_dc")
ax_dc.set_xlim(wavel_vis.min(), wavel_vis.max())
ax_dc.set_ylim(dc_vis.min(), 0.5)

ax_qe = fig.add_subplot(gs[0, 1])
_plot_marginalized_panel(
    ax_qe,
    wavel_vis,
    qe_list,
    snr_wavel_qe,
    "Wavelength (um)",
    "QE",
    f"Marginalized over DC (max S/N, contour at {s2n_min:g})",
)
for detector in detectors_dict.values():
    detector.plot(ax_qe, mode="wavel_qe")
ax_qe.set_xlim(wavel_vis.min(), wavel_vis.max())
ax_qe.set_ylim(qe_list.min(), 1.0)

fig.suptitle(f"Acceptable S/N region (S/N >= {s2n_min:g})", y=1.02)
fig.tight_layout()

# Place labels after layout is fixed so overlap checks use final display coords.
bold_names = _detectors_visible_on_both_panels(ax_dc, ax_qe, detectors_dict.values())
_place_detector_labels(ax_dc, detectors_dict.values(), mode="wavel_dc", bold_names=bold_names)
_place_detector_labels(ax_qe, detectors_dict.values(), mode="wavel_qe", bold_names=bold_names)

out_path = "/Users/eckhartspalding/Downloads/junk_acceptable_s2n_region_3d_isosurface.png"
plt.savefig(out_path, bbox_inches="tight", dpi=200)
print(f"Saved plot to {out_path}")

# Annotated 3D Plotly view, saved as a separate high-res PNG.
_save_annotated_volume_png(
    snr_plot,
    wavel_vis,
    qe_list,
    dc_vis,
    s2n_min,
    detectors_dict,
    z_max=0.5,
    camera_zoom=camera_zoom,
    camera_angles_deg=camera_angles_deg,
)

#plt.show()
