"# Masked Spatiotemporal GCN - Autism Spectrum Disorder Diagnosis from fMRI Data

> Official PyTorch Implementation - **Masked Spatiotemporal GCN**

> Authors: Anonymous (Based on AAAI 2026 Submission Paper)


## Project Overview

Masked Spatiotemporal GCN is a deep learning model for Autism Spectrum Disorder (ASD) diagnosis based on resting-state fMRI data. This model addresses three major challenges in ASD diagnosis: capturing brain relational patterns, understanding complex brain node relationships, and extracting common features from heterogeneous data.

### Core Problems Solved
1. **Capturing Brain Relational Patterns** - Using graph convolution and spatiotemporal attention mechanisms to focus on key brain regions.
2. **Understanding Brain Node Relationships** - Designing a masked reconstruction task to learn inter-node relationships.
3. **Extracting Features from Heterogeneous Data** - Proposing intra-group and inter-group difference and connection optimization loss functions.

### Technical Architecture

#### Core Components
- **GCN-STA (Graph Convolutional Network with Spatiotemporal Attention)** - Graph convolutional network combined with spatiotemporal attention to capture brain network structures.
- **MRM (Masked Reconstruction Module)** - Masked reconstruction module to learn complex relationships between brain nodes.
- **FOM (Feature Optimization Module)** - Feature optimization module, including intra-group homogeneity loss, inter-group heterogeneity loss, and connection refinement loss.

#### Model Features
- Supports spatiotemporal attention mechanisms.
- Combined loss functions to balance classification and regularization.

## Quick Start

### Environment Requirements
```bash
# Main dependencies
torch >= 1.8.0
numpy
scikit-learn
nilearn
pandas
matplotlib
scipy
pyyaml
tensorboard
```

### Data Preparation

#### ABIDE Dataset
1. Download data:
```bash
cd util/abide/
python 01-fetch_data.py --root_path /path/to/save/ --id_file_path subject_IDs.txt --download True
```

2. Process data:
```bash
python 02-process_data.py --root_path /path/to/save/ --id_file_path subject_IDs.txt
```

3. Generate final dataset:
```bash
python 03-generate_abide_dataset.py --root_path /path/to/save/
```

### Model Training

#### Basic Training
```bash
python main.py --config_filename setting/abide_RGTNet.yaml
```



model:
  type: MaskedSpatiotemporalGCN
  gcn_layers: 4
  attention_heads: 4
  embedding_dim: 8
  masking_prob: 0.2

train:
  lr: 1.0e-4
  epochs: 500
  alpha: 0.01
  beta: 0.001
  gamma: 0.1
  log_folder: /result
```

## Performance Results

### ABIDE Dataset Results
Results from 5-fold cross-validation using CC200, CC400, AAL atlases:

| Atlas | ACC(%) | AUC(%) | SPE(%) | SEN(%) |
|-------|--------|--------|--------|--------|
| CC200 | 70.18 | 75.32 | 70.00 | 81.08 |
| CC400 | 77.23 | 75.94 | 71.88 | 81.82 |
| AAL   | 79.79 | 73.94 | 67.88 | 85.76 |

The model outperforms existing methods on multiple metrics.

## Project Structure

```
RGTNet-main/
├── main.py                 # Main entry point
├── train.py               # Training logic
├── dataloader.py          # Data loader
├── model/                 # Model components
│   ├── GraphTransformer.py    # Graph transformation related (extensible for new models)
│   ├── GTEncoder.py           # Encoder
│   ├── CBAM.py               # Attention mechanism
│   ├── S2Attention.py        # Attention mechanism
│   ├── EMSA.py              # Attention mechanism
│   ├── CoordAttention.py     # Attention mechanism
│   ├── SimplifiedSelfAttention.py # Attention mechanism
│   ├── convGRU.py           # GRU component
│   └── PE.py               # Positional encoding
├── setting/               # Configuration files
│   └── abide_RGTNet.yaml  # Configuration
├── util/                  # Utilities
│   ├── abide/             # Data processing
│   ├── analysis/          # Analysis tools
│   ├── logger.py          # Logging
│   ├── loss.py            # Loss functions
│   ├── meter.py           # Metrics
│   └── prepossess.py      # Preprocessing
└── logs/                  # Logs
```


MIT License.

---



