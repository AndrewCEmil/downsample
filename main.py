from argparse import ArgumentParser, ArgumentTypeError
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from algorithms import algorithm_names, run_algorithm


RENDER_CELL_SIZE = 32
RENDER_FILENAME = "render.png"
PALETTE_FILENAME = "palette.csv"
GRID_FILENAME = "grid.csv"


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArgumentTypeError("must be an integer") from exc

    if parsed < 1:
        raise ArgumentTypeError("must be greater than zero")

    return parsed


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.float32) / 255.0


def rgb_float_to_uint8(colors: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)


def render_grid_image(
    palette: np.ndarray,
    grid: np.ndarray,
    cell_size: int = RENDER_CELL_SIZE,
) -> Image.Image:
    grid_colors = rgb_float_to_uint8(palette[grid])
    image = Image.fromarray(grid_colors)
    return image.resize(
        (grid.shape[1] * cell_size, grid.shape[0] * cell_size),
        Image.Resampling.NEAREST,
    )


def write_palette_csv(path: Path, palette: np.ndarray) -> None:
    colors = rgb_float_to_uint8(palette)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "r", "g", "b", "hex"])

        for index, (red, green, blue) in enumerate(colors):
            writer.writerow(
                [
                    index,
                    int(red),
                    int(green),
                    int(blue),
                    f"#{red:02X}{green:02X}{blue:02X}",
                ]
            )


def write_grid_csv(path: Path, grid: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(grid.tolist())


def write_outputs(output_directory_path: Path, palette: np.ndarray, grid: np.ndarray) -> None:
    render_grid_image(palette, grid).save(output_directory_path / RENDER_FILENAME)
    write_palette_csv(output_directory_path / PALETTE_FILENAME, palette)
    write_grid_csv(output_directory_path / GRID_FILENAME, grid)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Downsample an image into a structured color grid.")
    parser.add_argument(
        "input_image_path",
        type=Path,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "output_directory_path",
        type=Path,
        help="Directory for generated output files.",
    )
    parser.add_argument(
        "rows",
        type=positive_int,
        help="Number of rows in the output grid.",
    )
    parser.add_argument(
        "columns",
        type=positive_int,
        help="Number of columns in the output grid.",
    )
    parser.add_argument(
        "colors",
        type=positive_int,
        help="Number of colors to use in the output palette.",
    )
    parser.add_argument(
        "algorithm",
        choices=algorithm_names(),
        help="Downsampling algorithm to use.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_data = load_rgb_image(args.input_image_path)
    result = run_algorithm(
        args.algorithm,
        image_data,
        args.colors,
        args.rows,
        args.columns,
    )

    args.output_directory_path.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_directory_path, result.palette, result.grid)
    print(
        "Processed "
        f"{args.input_image_path} -> {args.output_directory_path} "
        f"({args.rows} rows, {args.columns} columns, {args.colors} colors, "
        f"{args.algorithm} algorithm, {image_data.shape} image, "
        f"{result.palette.shape} palette, {result.grid.shape} grid; "
        f"wrote {RENDER_FILENAME}, {PALETTE_FILENAME}, {GRID_FILENAME})"
    )


if __name__ == "__main__":
    main()
