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


def sample_kmeans_rgb_palette(
    image_data: np.ndarray,
    colors: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    points = representative_grid_colors(image_data, rows, columns).reshape(-1, 3)

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


def sample_median_cut_palette(
    image_data: np.ndarray,
    colors: int,
    rows: int,
    columns: int,
) -> np.ndarray:
    raise NotImplementedError("median-cut palette sampling is not implemented yet")


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


ALGORITHMS: dict[str, Algorithm] = {
    "kmeans-rgb": Algorithm(sample_kmeans_rgb_palette, assign_nearest_rgb_grid),
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
