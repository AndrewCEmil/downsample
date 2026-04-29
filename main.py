from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import numpy as np
from PIL import Image

from algorithms import algorithm_names, run_algorithm


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
    print(
        "Processed "
        f"{args.input_image_path} -> {args.output_directory_path} "
        f"({args.rows} rows, {args.columns} columns, {args.colors} colors, "
        f"{args.algorithm} algorithm, {image_data.shape} image, "
        f"{result.palette.shape} palette, {result.grid.shape} grid)"
    )


if __name__ == "__main__":
    main()
