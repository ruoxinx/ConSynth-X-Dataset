#!/usr/bin/env python3
"""
Merge batch Arrow files into a single Arrow file
"""
import argparse
from pathlib import Path
import pyarrow as pa


def load_arrow_file(path):
    """Load Arrow IPC file"""
    with open(path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
    return table


def save_arrow_file(table, path):
    """Save table to Arrow IPC file"""
    with open(path, 'wb') as f:
        writer = pa.ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()


def main():
    parser = argparse.ArgumentParser(description='Merge batch Arrow files')
    parser.add_argument('--input-dir', '-i', required=True,
                       help='Directory containing batch_*.arrow files')
    parser.add_argument('--output', '-o', required=True,
                       help='Output merged Arrow file path')
    parser.add_argument('--pattern', default='batch_*.arrow',
                       help='Glob pattern for batch files (default: batch_*.arrow)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    # Find all batch files
    batch_files = sorted(input_dir.glob(args.pattern))
    
    if not batch_files:
        print(f"❌ No files matching '{args.pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(batch_files)} batch files")
    
    # Load and concatenate all tables
    tables = []
    total_rows = 0
    
    for batch_file in batch_files:
        print(f"  Loading {batch_file.name}...", end=" ")
        table = load_arrow_file(batch_file)
        print(f"{len(table)} rows")
        tables.append(table)
        total_rows += len(table)
    
    # Concatenate
    print(f"\nMerging {len(tables)} tables...")
    merged_table = pa.concat_tables(tables)
    
    print(f"Total rows: {len(merged_table)}")
    print(f"Columns: {merged_table.column_names}")
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_arrow_file(merged_table, output_path)
    
    print(f"\n✅ Done!")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Total samples: {len(merged_table)}")


if __name__ == '__main__':
    main()


# Usage:
# python merge_arrow_batches.py -i output/construction_site_test/night -o output/construction_site_test/night_merged.arrow
