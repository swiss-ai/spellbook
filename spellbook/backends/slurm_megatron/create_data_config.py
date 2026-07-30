# From SwissAI-Megatron

import argparse
import os
from pathlib import Path


def create_data_prefix(
    list_of_paths: list[str], *, follow_symlinks: bool = False
) -> list[str]:
    list_of_bin_files: list[str] = []
    # Select all .bin files
    for path in list_of_paths:
        path_to_files = [
            os.path.join(dp, f)
            for dp, _, fn in os.walk(
                os.path.expanduser(path), followlinks=follow_symlinks
            )
            for f in fn
        ]
        list_of_bin_files.extend(
            [
                raw_file
                for raw_file in path_to_files
                if Path(raw_file).suffix.lower().endswith(".bin")
            ]
        )

    return sorted({
        bin_file[:-4] for bin_file in list_of_bin_files
    })  # NOTE(tj.solergibert) Delete .bin extension to have file prefixes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        required=True,
        help="Comma separated list of paths to generate the config from. e.g. -p /path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinked directories while discovering .bin shards.",
    )
    args = parser.parse_args()

    paths = [x.strip() for x in args.paths.split(",")]
    data_prefix = create_data_prefix(paths, follow_symlinks=args.follow_symlinks)
    print(*data_prefix, sep=" ")
