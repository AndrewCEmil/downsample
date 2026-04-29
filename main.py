from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArgumentTypeError("must be an integer") from exc

    if parsed < 1:
        raise ArgumentTypeError("must be greater than zero")

    return parsed


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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.output_directory_path.mkdir(parents=True, exist_ok=True)
    print(
        "Ready to process "
        f"{args.input_image_path} -> {args.output_directory_path} "
        f"({args.rows} rows, {args.columns} columns, {args.colors} colors)"
    )


if __name__ == "__main__":
    main()
