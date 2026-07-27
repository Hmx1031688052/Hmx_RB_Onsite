import os
import time

import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def _as_points(points):
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return None
    return arr[:, :2]


def _expand_bounds(bounds, margin_ratio=0.12, min_margin=20.0):
    x_min, x_max, y_min, y_max = bounds
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    margin = max(width, height) * margin_ratio
    margin = max(margin, min_margin)
    return x_min - margin, x_max + margin, y_min - margin, y_max + margin


def _update_bounds(bounds, points):
    pts = _as_points(points)
    if pts is None:
        return bounds
    x_min = float(np.nanmin(pts[:, 0]))
    x_max = float(np.nanmax(pts[:, 0]))
    y_min = float(np.nanmin(pts[:, 1]))
    y_max = float(np.nanmax(pts[:, 1]))
    if bounds is None:
        return x_min, x_max, y_min, y_max
    return (
        min(bounds[0], x_min),
        max(bounds[1], x_max),
        min(bounds[2], y_min),
        max(bounds[3], y_max),
    )


def _plot_map(ax, lane_info):
    if lane_info is None or not hasattr(lane_info, "discretelanes"):
        return None

    bounds = None
    for lane in lane_info.discretelanes:
        left = _as_points(getattr(lane, "left_vertices", None))
        right = _as_points(getattr(lane, "right_vertices", None))
        center = _as_points(getattr(lane, "center_vertices", None))

        if left is not None and right is not None:
            polygon = np.vstack([left, right[::-1]])
            ax.fill(
                polygon[:, 0],
                polygon[:, 1],
                color="#e8e8e8",
                alpha=0.65,
                linewidth=0,
                zorder=0,
            )
            ax.plot(left[:, 0], left[:, 1], color="#b5b5b5", linewidth=0.35, zorder=1)
            ax.plot(right[:, 0], right[:, 1], color="#b5b5b5", linewidth=0.35, zorder=1)
            bounds = _update_bounds(bounds, polygon)

        if center is not None:
            ax.plot(center[:, 0], center[:, 1], color="white", linewidth=0.45, alpha=0.75, zorder=2)
            bounds = _update_bounds(bounds, center)

    return bounds


def _plot_routes(ax, route_dict, goal_lane):
    if not route_dict:
        return None

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#17becf",
    ]
    bounds = None
    for idx, (route_id, route) in enumerate(route_dict.items()):
        points = _as_points(route.get("center_vertices"))
        if points is None:
            continue

        is_goal_route = str(route_id) == str(goal_lane)
        color = "#d62728" if is_goal_route else colors[idx % len(colors)]
        linewidth = 3.0 if is_goal_route else 2.0
        label = "goal route {}".format(route_id) if is_goal_route else "route {}".format(route_id)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=0.95,
            label=label,
            zorder=5 if is_goal_route else 4,
        )
        ax.scatter(points[:: max(len(points) // 80, 1), 0], points[:: max(len(points) // 80, 1), 1],
                   s=5, color=color, alpha=0.65, zorder=6)
        bounds = _update_bounds(bounds, points)

    return bounds


def _plot_named_point(ax, xy, label, color, marker, size):
    x, y = float(xy[0]), float(xy[1])
    ax.scatter([x], [y], s=size, marker=marker, color=color, edgecolor="black", linewidth=0.6, zorder=9)
    ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold", zorder=10)


def _plot_waypoints(ax, waypoints):
    bounds = None
    for idx, point in enumerate(waypoints or [], start=1):
        if point is None or len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        ax.scatter([x], [y], s=32, marker="D", color="#202020", edgecolor="white", linewidth=0.6, zorder=8)
        ax.annotate("WP{}".format(idx), (x, y), xytext=(5, -10), textcoords="offset points", fontsize=8, zorder=10)
        bounds = _update_bounds(bounds, [[x, y]])
    return bounds


def save_global_plan_visualization(
    lane_info,
    route_dict,
    map_file,
    start_xy,
    goal_xy,
    goal_lane=None,
    waypoints=None,
    output_dir=None,
    episode_index=None,
    full_map=False,
):
    if output_dir is None:
        output_dir = os.environ.get("E2E_VIS_GLOBAL_PLAN_DIR")
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(__file__), "global_plan_vis")
    os.makedirs(output_dir, exist_ok=True)

    fig = Figure(figsize=(12, 9), dpi=150)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    map_bounds = _plot_map(ax, lane_info)
    route_bounds = _plot_routes(ax, route_dict, goal_lane)
    waypoint_bounds = _plot_waypoints(ax, waypoints)

    _plot_named_point(ax, start_xy, "START", "#2ca02c", "o", 70)
    _plot_named_point(ax, goal_xy, "GOAL", "#d62728", "*", 130)

    focus_bounds = route_bounds
    focus_bounds = _update_bounds(focus_bounds, [start_xy, goal_xy])
    if waypoint_bounds is not None:
        focus_bounds = _update_bounds(focus_bounds, [[waypoint_bounds[0], waypoint_bounds[2]], [waypoint_bounds[1], waypoint_bounds[3]]])

    if full_map and map_bounds is not None:
        plot_bounds = map_bounds
    else:
        plot_bounds = focus_bounds or map_bounds

    if plot_bounds is not None:
        x_min, x_max, y_min, y_max = _expand_bounds(plot_bounds)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    route_count = len(route_dict or {})
    title_map = os.path.basename(str(map_file))
    ax.set_title("Global Plan: {}".format(title_map), fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d0d0d0", linewidth=0.4, alpha=0.65)

    info = [
        "routes: {}".format(route_count),
        "goal route: {}".format(goal_lane),
        "start: ({:.2f}, {:.2f})".format(float(start_xy[0]), float(start_xy[1])),
        "goal: ({:.2f}, {:.2f})".format(float(goal_xy[0]), float(goal_xy[1])),
    ]
    ax.text(
        0.01,
        0.99,
        "\n".join(info),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc", "linewidth": 0.5},
        zorder=20,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:8], labels[:8], loc="lower right", fontsize=8, framealpha=0.9)

    prefix = "episode_{:04d}".format(episode_index) if episode_index is not None else time.strftime("%Y%m%d_%H%M%S")
    safe_map = os.path.splitext(title_map)[0].replace(" ", "_")
    output_path = os.path.join(output_dir, "{}_{}.png".format(prefix, safe_map))
    fig.tight_layout()
    fig.savefig(output_path)
    return output_path
