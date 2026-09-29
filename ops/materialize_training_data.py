"""Convert bundled frozen Parquet splits into OAT's local DatasetDict format."""
import argparse
from pathlib import Path
from datasets import DatasetDict
from modebench.data import load_split

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1]/'data')
p.add_argument('--config', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error('output already exists; use a fresh directory')
train = load_split(a.data_root, a.config, 'train')
test = load_split(a.data_root, a.config, 'eval')
DatasetDict({'train':train}).save_to_disk(str(a.output/'train'))
DatasetDict({'multi_answer':test}).save_to_disk(str(a.output/'eval'))
