# DOSC
DISC - Artificial toy dataset for benchmarking attribution methods

## Installation
```console
conda create -n disc python=3.6
conda activate disc
pip install -r requirements.txt
pip install .
```

## Usage
Create toy dataset via:
```console
python create_all.py
```

For training via pytorch use the provided dataloader at `disc/loader.py`
