#!/usr/bin/env python3
"""Split recommendation-only pretrain data into multiple parquet shards.

Usage:
  python test/split_rec_only.py \
    --rec_data_path output/pretrain.parquet \
    --output_dir output/split_data_pretrain_rec \
    --max_rows 1000 \
    --engine pyarrow
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from data.scripts.split_data import find_parquet_files, load_all_parquet_files, split_dataframe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Split recommendation-only pretrain data into multiple files'
    )
    parser.add_argument(
        '--rec_data_path',
        type=str,
        required=True,
        help='Recommendation data path (directory or file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory path'
    )
    parser.add_argument(
        '--max_rows',
        type=int,
        default=1000,
        help='Maximum number of rows per file (default: 1000)'
    )
    parser.add_argument(
        '--engine',
        choices=['pyarrow', 'fastparquet'],
        default='pyarrow',
        help='Parquet processing engine (default: pyarrow)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not recursively search for files in subdirectories'
    )
    args = parser.parse_args()

    if args.max_rows <= 0:
        logger.error(f"max_rows must be greater than 0, current value: {args.max_rows}")
        sys.exit(1)

    try:
        logger.info("=" * 60)
        logger.info("Step 1: Finding recommendation data files...")
        rec_data_path = Path(args.rec_data_path)
        if rec_data_path.is_file():
            rec_data_files = [str(rec_data_path)]
        else:
            rec_data_files = find_parquet_files(
                args.rec_data_path,
                recursive=not args.no_recursive
            )
        logger.info(f"Found {len(rec_data_files)} recommendation data files")

        logger.info("=" * 60)
        logger.info("Step 2: Loading recommendation data...")
        rec_data_df = load_all_parquet_files(rec_data_files, engine=args.engine)

        if len(rec_data_df) == 0:
            logger.error("No recommendation data loaded")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("Step 3: Splitting data...")
        output_files = split_dataframe(
            rec_data_df,
            max_rows=args.max_rows,
            output_dir=args.output_dir,
            prefix="part"
        )

        logger.info("=" * 60)
        logger.info("Step 4: Generating file list JSON...")
        output_dir_path = Path(args.output_dir)
        json_file_path = output_dir_path / "file_list.json"
        file_list = [str(Path(f).absolute()) for f in output_files]

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(file_list, f, indent=2, ensure_ascii=False)

        logger.info(f"File list saved to: {json_file_path} ({len(file_list)} files)")
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Input files: {len(rec_data_files)}")
        logger.info(f"Total data rows: {len(rec_data_df)}")
        logger.info(f"Output files: {len(output_files)}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"File list JSON: {json_file_path}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
