import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import render_grid_image, write_grid_csv, write_palette_csv


class OutputTests(unittest.TestCase):
    def test_render_grid_image_expands_grid_colors(self) -> None:
        palette = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        grid = np.array([[0, 1], [2, 3]], dtype=np.int64)

        image = render_grid_image(palette, grid, cell_size=2)
        pixels = np.asarray(image)

        self.assertEqual(image.size, (4, 4))
        np.testing.assert_array_equal(
            pixels[0:2, 0:2],
            np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            pixels[0:2, 2:4],
            np.full((2, 2, 3), [0, 255, 0], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            pixels[2:4, 0:2],
            np.full((2, 2, 3), [0, 0, 255], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            pixels[2:4, 2:4],
            np.full((2, 2, 3), [255, 255, 255], dtype=np.uint8),
        )

    def test_write_palette_csv_writes_rgb_and_hex_values(self) -> None:
        palette = np.array(
            [
                [1.0, 0.0, 0.5],
                [0.25, 0.5, 0.75],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "palette.csv"

            write_palette_csv(path, palette)

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "index,r,g,b,hex\n"
                "0,255,0,128,#FF0080\n"
                "1,64,128,191,#4080BF\n",
            )

    def test_write_grid_csv_writes_palette_indexes(self) -> None:
        grid = np.array([[0, 2, 1], [1, 0, 2]], dtype=np.int64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.csv"

            write_grid_csv(path, grid)

            self.assertEqual(path.read_text(encoding="utf-8"), "0,2,1\n1,0,2\n")


if __name__ == "__main__":
    unittest.main()
