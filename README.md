# Graph Attention

Graph attention mechanism experiments and implementation.

## Installation

### Using pixi

```bash
pixi install
pixi shell
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Training is managed via [Hydra](https://hydra.cc/). You can run experiments and override configuration parameters from the command line.

### Training

```bash
# Default configuration
python train.py

# Run a specific experiment configuration
python train.py +experiment=finetune_resnet

# Override parameters
python train.py model=vit_tiny dataset=cifar100 training.epochs=50
```

Configuration files are located in the `configs/` directory.

### Scripts

You can also run standalone scripts located in the `scripts/` folder. These scripts do not use Hydra configuration.

```bash
python scripts/train_example_imagenet.py
```

## Development Workflow

-   **Experimentation**: Use the `scripts/` and `notebooks/` directories to play around, test ideas, and run one-off experiments. These are great for prototyping.
-   **Persistent Components**: If you develop something useful for the team (e.g., a new dataset, model architecture, or utility), please integrate it into the main project structure (e.g., `graph_attention/models`, `graph_attention/data`) and add a corresponding configuration in `configs/`.

