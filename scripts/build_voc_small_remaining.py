"""Filter soda_voc original arrow to only the 2000 image_ids NOT in existing soda_small."""
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path('/users/PGS0407/binben14/VietHuy/ConSynth-X')
ORIG = ROOT / 'augmentation_data/soda_voc/original/soda_voc_original_first3000.arrow'
EXISTING = ROOT / 'augmentation_data/soda_voc/small/soda_small.arrow'
OUT = ROOT / 'augmentation_data/soda_voc/small/_remaining_2000_source.arrow'


def main():
    src_e = pa.memory_map(str(EXISTING), 'r'); r_e = ipc.open_stream(src_e)
    have = set()
    for b in r_e:
        have.update(b.column('image_id').to_pylist())
    print(f'existing small: {len(have)} ids')

    src_o = pa.memory_map(str(ORIG), 'r'); r_o = ipc.open_stream(src_o)
    tables = []
    for b in r_o:
        ids = b.column('image_id').to_pylist()
        keep_idx = [i for i, iid in enumerate(ids) if iid not in have]
        if not keep_idx:
            continue
        sub = b.take(keep_idx)
        tables.append(pa.Table.from_batches([sub]))
    out_table = pa.concat_tables(tables)
    print(f'remaining to generate: {out_table.num_rows} rows, schema={out_table.schema.names}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(OUT), 'wb') as sink:
        with ipc.new_stream(sink, out_table.schema) as w:
            for batch in out_table.to_batches():
                w.write_batch(batch)
    print(f'-> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
