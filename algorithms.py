from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np


PaletteSampler: TypeAlias = Callable[[np.ndarray, int, int, int], np.ndarray]
GridAssigner: TypeAlias = Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]


@dataclass(frozen=True)
class DownsampleResult:
    palette: np.ndarray
    grid: np.ndarray


@dataclass(frozen=True)
class Algorithm:
    sample_palette: PaletteSampler
    assign_grid: GridAssigner


KMEANS_MAX_ITERATIONS = 100
KMEANS_RANDOM_SEED = 0
CIELAB_DELTA = 6 / 29
XYZ_D65_WHITE = np.array([0.95047, 1.0, 1.08883])
RGB_TO_XYZ_MATRIX = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
XYZ_TO_RGB_MATRIX = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)


def _srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    return np.where(
        colors <= 0.04045,
        colors / 12.92,
        ((colors + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    nonnegative_colors = np.maximum(colors, 0.0)
    return np.where(
        nonnegative_colors <= 0.0031308,
        12.92 * nonnegative_colors,
        1.055 * (nonnegative_colors ** (1 / 2.4)) - 0.055,
    )


def rgb_to_lab(colors: np.ndarray) -> np.ndarray:
    linear_rgb = _srgb_to_linear(np.clip(colors, 0.0, 1.0))
    xyz = linear_rgb @ RGB_TO_XYZ_MATRIX.T
    normalized_xyz = xyz / XYZ_D65_WHITE

    delta_cubed = CIELAB_DELTA**3
    transformed = np.where(
        normalized_xyz > delta_cubed,
        np.cbrt(normalized_xyz),
        normalized_xyz / (3 * CIELAB_DELTA**2) + 4 / 29,
    )

    return np.stack(
        [
            116 * transformed[..., 1] - 16,
            500 * (transformed[..., 0] - transformed[..., 1]),
            200 * (transformed[..., 1] - transformed[..., 2]),
        ],
        axis=-1,
    )


def lab_to_rgb(colors: np.ndarray) -> np.ndarray:
    lightness = colors[..., 0]
    fy = (lightness + 16) / 116
    fx = fy + colors[..., 1] / 500
    fz = fy - colors[..., 2] / 200
    transformed = np.stack([fx, fy, fz], axis=-1)

    xyz = XYZ_D65_WHITE * np.where(
        transformed > CIELAB_DELTA,
        transformed**3,
        3 * CIELAB_DELTA**2 * (transformed - 4 / 29),
    )
    linear_rgb = xyz @ XYZ_TO_RGB_MATRIX.T
    return np.clip(_linear_to_srgb(linear_rgb), 0.0, 1.0)


def representative_grid_colors(
    image_data: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    cell_colors = np.empty((rows, columns, 3), dtype=image_data.dtype)

    for row in range(rows):
        row_start = int(row * image_data.shape[0] / rows)
        row_end = int((row + 1) * image_data.shape[0] / rows)

        for column in range(columns):
            column_start = int(column * image_data.shape[1] / columns)
            column_end = int((column + 1) * image_data.shape[1] / columns)
            cell = image_data[row_start:row_end, column_start:column_end]

            if cell.size:
                cell_colors[row, column] = cell.reshape(-1, 3).mean(axis=0)
            else:
                source_row = min(
                    int((row + 0.5) * image_data.shape[0] / rows),
                    image_data.shape[0] - 1,
                )
                source_column = min(
                    int((column + 0.5) * image_data.shape[1] / columns),
                    image_data.shape[1] - 1,
                )
                cell_colors[row, column] = image_data[source_row, source_column]

    return cell_colors


def _sample_kmeans_palette(points: np.ndarray, colors: int) -> np.ndarray:
    if colors > len(points):
        raise ValueError("colors must be less than or equal to rows * columns")

    rng = np.random.default_rng(KMEANS_RANDOM_SEED)
    initial_indices = rng.choice(len(points), size=colors, replace=False)
    centers = points[initial_indices].copy()
    labels = np.full(len(points), -1, dtype=np.int64)

    for _ in range(KMEANS_MAX_ITERATIONS):
        distances = np.sum(
            (points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        new_labels = np.argmin(distances, axis=1)
        closest_distances = distances[np.arange(len(points)), new_labels]
        new_centers = centers.copy()

        for color_index in range(colors):
            cluster_points = points[new_labels == color_index]
            if len(cluster_points):
                new_centers[color_index] = cluster_points.mean(axis=0)
            else:
                farthest_point_index = int(np.argmax(closest_distances))
                new_centers[color_index] = points[farthest_point_index]
                closest_distances[farthest_point_index] = -1.0

        converged = np.array_equal(new_labels, labels) and np.allclose(
            new_centers,
            centers,
        )
        labels = new_labels
        centers = new_centers

        if converged:
            break

    return centers


def sample_kmeans_rgb_palette(
    image_data: np.ndarray,
    colors: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    points = representative_grid_colors(image_data, rows, columns).reshape(-1, 3)
    return _sample_kmeans_palette(points, colors)


def sample_kmeans_lab_palette(
    image_data: np.ndarray,
    colors: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    rgb_points = representative_grid_colors(image_data, rows, columns).reshape(-1, 3)
    lab_points = rgb_to_lab(rgb_points)
    lab_palette = _sample_kmeans_palette(lab_points, colors)
    return lab_to_rgb(lab_palette).astype(image_data.dtype, copy=False)


def sample_median_cut_palette(
    image_data: np.ndarray,
    colors: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    points = representative_grid_colors(image_data, rows, columns).reshape(-1, 3)

    if colors > len(points):
        raise ValueError("colors must be less than or equal to rows * columns")

    boxes = [points]

    while len(boxes) < colors:
        split_index = max(
            (index for index, box in enumerate(boxes) if len(box) > 1),
            key=lambda index: (
                float(np.ptp(boxes[index], axis=0).max()),
                len(boxes[index]),
                -index,
            ),
        )
        box = boxes.pop(split_index)
        split_channel = int(np.argmax(np.ptp(box, axis=0)))
        sorted_box = box[np.argsort(box[:, split_channel], kind="mergesort")]
        midpoint = len(sorted_box) // 2

        boxes.append(sorted_box[:midpoint])
        boxes.append(sorted_box[midpoint:])

    return np.array([box.mean(axis=0) for box in boxes], dtype=points.dtype)


def sample_pillow_palette(
    image_data: np.ndarray,
    colors: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    raise NotImplementedError("pillow palette sampling is not implemented yet")


def assign_nearest_rgb_grid(
    image_data: np.ndarray,
    palette: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    grid = np.empty((rows, columns), dtype=np.int64)
    cell_colors = representative_grid_colors(image_data, rows, columns)

    for row in range(rows):
        for column in range(columns):
            distances = np.sum((palette - cell_colors[row, column]) ** 2, axis=1)
            grid[row, column] = int(np.argmin(distances))

    return grid


def assign_nearest_lab_grid(
    image_data: np.ndarray,
    palette: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    grid = np.empty((rows, columns), dtype=np.int64)
    cell_colors = representative_grid_colors(image_data, rows, columns)
    lab_palette = rgb_to_lab(palette)
    lab_cell_colors = rgb_to_lab(cell_colors)

    for row in range(rows):
        for column in range(columns):
            distances = np.sum(
                (lab_palette - lab_cell_colors[row, column]) ** 2,
                axis=1,
            )
            grid[row, column] = int(np.argmin(distances))

    return grid


ALGORITHMS: dict[str, Algorithm] = {
    "kmeans-rgb": Algorithm(sample_kmeans_rgb_palette, assign_nearest_rgb_grid),
    "kmeans-lab": Algorithm(sample_kmeans_lab_palette, assign_nearest_lab_grid),
    "median-cut": Algorithm(sample_median_cut_palette, assign_nearest_rgb_grid),
    "pillow": Algorithm(sample_pillow_palette, assign_nearest_rgb_grid),
}


def algorithm_names() -> tuple[str, ...]:
    return tuple(ALGORITHMS)


def run_algorithm(
    name: str,
    image_data: np.ndarray,
    colors: int,
    rows: int,
    columns: int,
) -> DownsampleResult:
    algorithm = ALGORITHMS[name]
    palette = algorithm.sample_palette(image_data, colors, rows, columns)
    grid = algorithm.assign_grid(image_data, palette, rows, columns)
    return DownsampleResult(palette=palette, grid=grid)
