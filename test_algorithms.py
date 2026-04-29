import unittest

import numpy as np

from algorithms import (
    algorithm_names,
    lab_to_rgb,
    rgb_to_lab,
    run_algorithm,
    sample_kmeans_lab_palette,
    sample_kmeans_rgb_palette,
    sample_median_cut_palette,
)


class KMeansRgbTests(unittest.TestCase):
    def test_solid_color_image_returns_matching_palette(self) -> None:
        image_data = np.full((1, 3, 3), [0.25, 0.5, 0.75], dtype=np.float32)

        palette = sample_kmeans_rgb_palette(image_data, colors=3, rows=1, columns=3)

        np.testing.assert_allclose(
            palette,
            np.full((3, 3), [0.25, 0.5, 0.75], dtype=np.float32),
        )

    def test_two_color_image_assigns_expected_grid_colors(self) -> None:
        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        image_data = np.array(
            [
                [red, red, blue, blue],
                [red, red, blue, blue],
            ],
            dtype=np.float32,
        )

        result = run_algorithm("kmeans-rgb", image_data, colors=2, rows=1, columns=2)
        assigned_colors = result.palette[result.grid]

        np.testing.assert_allclose(assigned_colors[0, 0], red)
        np.testing.assert_allclose(assigned_colors[0, 1], blue)

    def test_colors_must_not_exceed_grid_cells(self) -> None:
        image_data = np.zeros((2, 2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "rows \\* columns"):
            sample_kmeans_rgb_palette(image_data, colors=2, rows=1, columns=1)

    def test_kmeans_rgb_is_deterministic(self) -> None:
        image_data = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )

        first = run_algorithm("kmeans-rgb", image_data, colors=3, rows=2, columns=2)
        second = run_algorithm("kmeans-rgb", image_data, colors=3, rows=2, columns=2)

        np.testing.assert_allclose(first.palette, second.palette)
        np.testing.assert_array_equal(first.grid, second.grid)


class ColorConversionTests(unittest.TestCase):
    def test_black_and_white_convert_to_expected_lab_values(self) -> None:
        colors = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        lab = rgb_to_lab(colors)

        np.testing.assert_allclose(lab[0], [0.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(lab[1], [100.0, 0.0, 0.0], atol=1e-3)

    def test_rgb_lab_rgb_round_trip_stays_close(self) -> None:
        colors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.25, 0.5, 0.75],
            ],
            dtype=np.float32,
        )

        converted = lab_to_rgb(rgb_to_lab(colors))

        np.testing.assert_allclose(converted, colors, atol=1e-5)


class KMeansLabTests(unittest.TestCase):
    def test_algorithm_is_registered(self) -> None:
        self.assertIn("kmeans-lab", algorithm_names())

    def test_solid_color_image_returns_matching_palette(self) -> None:
        image_data = np.full((1, 3, 3), [0.25, 0.5, 0.75], dtype=np.float32)

        palette = sample_kmeans_lab_palette(image_data, colors=3, rows=1, columns=3)

        np.testing.assert_allclose(
            palette,
            np.full((3, 3), [0.25, 0.5, 0.75], dtype=np.float32),
            atol=1e-5,
        )

    def test_two_color_image_assigns_expected_grid_colors(self) -> None:
        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        image_data = np.array(
            [
                [red, red, blue, blue],
                [red, red, blue, blue],
            ],
            dtype=np.float32,
        )

        result = run_algorithm("kmeans-lab", image_data, colors=2, rows=1, columns=2)
        assigned_colors = result.palette[result.grid]

        np.testing.assert_allclose(assigned_colors[0, 0], red, atol=1e-5)
        np.testing.assert_allclose(assigned_colors[0, 1], blue, atol=1e-5)

    def test_colors_must_not_exceed_grid_cells(self) -> None:
        image_data = np.zeros((2, 2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "rows \\* columns"):
            sample_kmeans_lab_palette(image_data, colors=2, rows=1, columns=1)

    def test_kmeans_lab_is_deterministic(self) -> None:
        image_data = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )

        first = run_algorithm("kmeans-lab", image_data, colors=3, rows=2, columns=2)
        second = run_algorithm("kmeans-lab", image_data, colors=3, rows=2, columns=2)

        np.testing.assert_allclose(first.palette, second.palette)
        np.testing.assert_array_equal(first.grid, second.grid)


class MedianCutTests(unittest.TestCase):
    def test_solid_color_image_returns_matching_palette(self) -> None:
        image_data = np.full((1, 3, 3), [0.25, 0.5, 0.75], dtype=np.float32)

        palette = sample_median_cut_palette(image_data, colors=3, rows=1, columns=3)

        np.testing.assert_allclose(
            palette,
            np.full((3, 3), [0.25, 0.5, 0.75], dtype=np.float32),
        )

    def test_two_color_image_assigns_expected_grid_colors(self) -> None:
        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        image_data = np.array(
            [
                [red, red, blue, blue],
                [red, red, blue, blue],
            ],
            dtype=np.float32,
        )

        result = run_algorithm("median-cut", image_data, colors=2, rows=1, columns=2)
        assigned_colors = result.palette[result.grid]

        np.testing.assert_allclose(assigned_colors[0, 0], red)
        np.testing.assert_allclose(assigned_colors[0, 1], blue)

    def test_colors_must_not_exceed_grid_cells(self) -> None:
        image_data = np.zeros((2, 2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "rows \\* columns"):
            sample_median_cut_palette(image_data, colors=2, rows=1, columns=1)

    def test_median_cut_is_deterministic(self) -> None:
        image_data = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )

        first = run_algorithm("median-cut", image_data, colors=3, rows=2, columns=2)
        second = run_algorithm("median-cut", image_data, colors=3, rows=2, columns=2)

        np.testing.assert_allclose(first.palette, second.palette)
        np.testing.assert_array_equal(first.grid, second.grid)


if __name__ == "__main__":
    unittest.main()
