from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np


PaletteSampler: TypeAlias = Callable[[np.ndarray, int], np.ndarray]
GridAssigner: TypeAlias = Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]


@dataclass(frozen=True)
class DownsampleResult:
    palette: np.ndarray
    grid: np.ndarray


@dataclass(frozen=True)
class Algorithm:
    sample_palette: PaletteSampler
    assign_grid: GridAssigner


def sample_kmeans_rgb_palette(image_data: np.ndarray, colors: int) -> np.ndarray:
    raise NotImplementedError("kmeans-rgb palette sampling is not implemented yet")


def sample_median_cut_palette(image_data: np.ndarray, colors: int) -> np.ndarray:
    raise NotImplementedError("median-cut palette sampling is not implemented yet")


def sample_pillow_palette(image_data: np.ndarray, colors: int) -> np.ndarray:
    raise NotImplementedError("pillow palette sampling is not implemented yet")


def assign_nearest_rgb_grid(
    image_data: np.ndarray,
    palette: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    grid = np.empty((rows, columns), dtype=np.int64)

    for row in range(rows):
        row_start = int(row * image_data.shape[0] / rows)
        row_end = int((row + 1) * image_data.shape[0] / rows)

        for column in range(columns):
            column_start = int(column * image_data.shape[1] / columns)
            column_end = int((column + 1) * image_data.shape[1] / columns)
            cell = image_data[row_start:row_end, column_start:column_end]

            if cell.size:
                cell_color = cell.reshape(-1, 3).mean(axis=0)
            else:
                source_row = min(
                    int((row + 0.5) * image_data.shape[0] / rows),
                    image_data.shape[0] - 1,
                )
                source_column = min(
                    int((column + 0.5) * image_data.shape[1] / columns),
                    image_data.shape[1] - 1,
                )
                cell_color = image_data[source_row, source_column]

            distances = np.sum((palette - cell_color) ** 2, axis=1)
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
    palette = algorithm.sample_palette(image_data, colors)
    grid = algorithm.assign_grid(image_data, palette, rows, columns)
    return DownsampleResult(palette=palette, grid=grid)
