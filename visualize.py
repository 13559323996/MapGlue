import matplotlib
import matplotlib.patheffects as peffects
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_images(image_list, titles=None, colormaps="gray", dpi=100, pad=0.5, auto_size=True):
    """
    Display a set of images horizontally.

    Args:
        image_list: List of images in either NumPy RGB (H, W, 3), 
                    PyTorch RGB (3, H, W) or grayscale (H, W) format.
        titles: List of titles for each image.
        colormaps: Colormap for grayscale images.
        dpi: Figure resolution.
        pad: Padding between images.
        auto_size: Whether the figure size should adapt to the images' aspect ratios.
    """
    # Convert torch.Tensor images to NumPy arrays in (H, W, 3) format.
    image_list = [
        img.permute(1, 2, 0).cpu().numpy()
        if (isinstance(img, torch.Tensor) and img.dim() == 3)
        else img
        for img in image_list
    ]
    num_imgs = len(image_list)
    if not isinstance(colormaps, (list, tuple)):
        colormaps = [colormaps] * num_imgs

    if auto_size:
        ratios = [im.shape[1] / im.shape[0] for im in image_list]  # width / height
    else:
        ratios = [4 / 3] * num_imgs
    fig_size = [sum(ratios) * 4.5, 4.5]
    fig, axes = plt.subplots(1, num_imgs, figsize=fig_size, dpi=dpi, gridspec_kw={"width_ratios": ratios})
    if num_imgs == 1:
        axes = [axes]
    for i in range(num_imgs):
        axes[i].imshow(image_list[i], cmap=plt.get_cmap(colormaps[i]))
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_axis_off()
        for spine in axes[i].spines.values():
            spine.set_visible(False)
        if titles:
            axes[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def draw_keypoints(keypoints, kp_color="lime", kp_size=4, ax_list=None, alpha_value=1.0):
    """
    Plot keypoints on existing images.

    Args:
        keypoints: List of ndarrays (N, 2) for each set of keypoints.
        kp_color: Color for keypoints, or list of colors for each set.
        kp_size: Size of keypoints.
        ax_list: List of axes to plot keypoints on; defaults to current figure's axes.
        alpha_value: Opacity for keypoints.
    """
    if not isinstance(kp_color, list):
        kp_color = [kp_color] * len(keypoints)
    if not isinstance(alpha_value, list):
        alpha_value = [alpha_value] * len(keypoints)
    if ax_list is None:
        ax_list = plt.gcf().axes
    for ax, pts, color, alpha in zip(ax_list, keypoints, kp_color, alpha_value):
        if isinstance(pts, torch.Tensor):
            pts = pts.cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=kp_size, linewidths=0, alpha=alpha)


def draw_matches(pts_left, pts_right, line_colors=None, line_width=1.5, endpoint_size=4, alpha_value=1.0, labels=None, axes_pair=None):
    """
    Draw matches between a pair of images.

    Args:
        pts_left, pts_right: Corresponding keypoints for the two images (N, 2).
        line_colors: Colors for each match line, either as a string or an RGB tuple.
                     If not provided, random colors will be generated.
        line_width: Width of the match lines.
        endpoint_size: Size of the endpoints (if 0, endpoints are not drawn).
        alpha_value: Opacity for the match lines.
        labels: Optional list of labels for each match.
        axes_pair: List of two axes [ax_left, ax_right] to plot the images; defaults to the first two axes in the current figure.
    """
    fig = plt.gcf()
    if axes_pair is None:
        axs = fig.axes
        ax_left, ax_right = axs[0], axs[1]
    else:
        ax_left, ax_right = axes_pair
    if isinstance(pts_left, torch.Tensor):
        pts_left = pts_left.cpu().numpy()
    if isinstance(pts_right, torch.Tensor):
        pts_right = pts_right.cpu().numpy()
    assert len(pts_left) == len(pts_right)
    if line_colors is None:
        line_colors = matplotlib.cm.hsv(np.random.rand(len(pts_left))).tolist()
    elif len(line_colors) > 0 and not isinstance(line_colors[0], (tuple, list)):
        line_colors = [line_colors] * len(pts_left)

    if line_width > 0:
        for i in range(len(pts_left)):
            connector = matplotlib.patches.ConnectionPatch(
                xyA=(pts_left[i, 0], pts_left[i, 1]),
                xyB=(pts_right[i, 0], pts_right[i, 1]),
                coordsA=ax_left.transData,
                coordsB=ax_right.transData,
                axesA=ax_left,
                axesB=ax_right,
                zorder=1,
                color=line_colors[i],
                linewidth=line_width,
                clip_on=True,
                alpha=alpha_value,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            connector.set_annotation_clip(True)
            fig.add_artist(connector)

    # Freeze axis autoscaling to prevent changes.
    ax_left.autoscale(enable=False)
    ax_right.autoscale(enable=False)

    if endpoint_size > 0:
        ax_left.scatter(pts_left[:, 0], pts_left[:, 1], c=line_colors, s=endpoint_size)
        ax_right.scatter(pts_right[:, 0], pts_right[:, 1], c=line_colors, s=endpoint_size)


def add_text(axis_idx, text, pos=(0.01, 0.99), font_size=15, txt_color="w", border_color="k", border_width=2, h_align="left", v_align="top"):
    """
    Add an annotation with an outline to a specified axis.

    Args:
        axis_idx: Index of the axis in the current figure where the annotation will be added.
        text: The annotation text.
        pos: Position of the annotation in axis coordinates (e.g., (0.01, 0.99)).
        font_size: Font size of the text.
        txt_color: Text color.
        border_color: Outline color (if None, no outline is applied).
        border_width: Width of the outline.
        h_align: Horizontal alignment (e.g., "left").
        v_align: Vertical alignment (e.g., "top").
    """
    current_ax = plt.gcf().axes[axis_idx]
    annotation = current_ax.text(
        *pos, text, fontsize=font_size, ha=h_align, va=v_align, color=txt_color, transform=current_ax.transAxes
    )
    if border_color is not None:
        annotation.set_path_effects([
            peffects.Stroke(linewidth=border_width, foreground=border_color),
            peffects.Normal(),
        ])
