# LACENet

LACENet is a lattice-aware graph neural network for crystal structure relaxation. It predicts relaxed atomic positions and lattice parameters from unrelaxed crystal structures.

## Installation

LACENet requires Python 3.9–3.12. Install PyTorch for your CUDA environment first, then install the project dependencies:



```bash
conda create -n lacenet python=3.10 -y
conda activate lacenet
python -m pip install -e ".[dev]"
```


## Data Preparation

The preprocessing script supports `xmno`, `mp`, `c2db`, `vdW`, and `jarvis` datasets. Each dataset must provide `train`, `val`, and `test` splits with paired unrelaxed and relaxed CIF files.

The data used in this study are available from the E3Relax-H2 project at https://github.com/guaguabujianle/E3RelaxH2.

Preprocess CIF pairs into LMDB files:

```bash
python -m preprocess.preprocess \
  --dataset xmno \
  --data_root /your/real/path/to/dataset \
  --num_workers 4 
```

The generated data are stored in:

```text
<data_root>/
├── train/*.lmdb
├── val/*.lmdb
└── test/*.lmdb
```

## Model Options

Two model variants are available:

- `Z`: updates the lattice graph before lattice-to-atom interaction.
- `O`: performs lattice-to-atom interaction before updating the lattice graph.

Two atom message-passing variants are available:

- `reference_gated` (aliases: `gated`, `mp1`), referred to as Adaptive Reference Direction Message Passing (ARD-MP) in the paper.
- `reference_masked` (aliases: `masked`, `mp2`), referred to as Masked Angular Message Passing (MA-MP) in the paper.

The same model options and architecture parameters must be used for training and evaluation.

## Training

```bash
python -m train.train_lacenet \
  --data_root /your/real/path/to/dataset \
  --dataset xmno \
  --model_variant Z \
  --message_passing reference_gated \
  --batch_size 32 \
  --num_workers 4 \
  --epochs 800 \
  --steps_per_epoch 800 \
  --checkpoint_dir ./checkpoints/xmno_Z_reference_gated \
  --device cuda:0 \
  --save_model
```

The training script writes the following files to `checkpoint_dir`:

- `model.pt`: best checkpoint based on validation position MAE.
- `last.pt`: latest training checkpoint.
- `interrupt.pt`: checkpoint saved after a keyboard interrupt.
- `validation_log.txt`: validation log.

The best checkpoint is only created when `--save_model` is enabled.

To resume training:

```bash
python -m train.train_lacenet \
  --data_root /your/real/path/to/dataset \
  --dataset xmno \
  --model_variant Z \
  --message_passing reference_gated \
  --checkpoint_dir ./checkpoints/xmno_Z_reference_gated \
  --resume_path ./checkpoints/xmno_Z_reference_gated/last.pt \
  --device cuda:0 \
  --save_model
```

A configurable Bash example is also available at `train/run_lacenet.sh`.

## Evaluation

```bash
python -m test.test_lacenet \
  --dataset xmno \
  --data_root /your/real/path/to/dataset \
  --model_path ./checkpoints/xmno_Z_reference_gated/model.pt \
  --model_variant Z \
  --message_passing reference_gated \
  --batch_size 8 \
  --num_workers 4 \
  --results_dir ./results \
  --device cuda:0
```

Evaluation saves per-structure metrics to:

```text
results/<dataset>_LACENet.csv
```

To export predicted relaxed structures, add `--save_structures`. CIF files will be written to:

```text
predicted_structures/<dataset>/
```

A configurable Bash example is available at `test/run_lacenet_test.sh`.

