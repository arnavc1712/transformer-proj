# GPT-2 Pretraining Project

This project implements a GPT-2 pretraining pipeline based on Andrej Karpathy's educational videos. It incorporates various advanced techniques to optimize training efficiency and performance.

## Features

- **GPT-2 Architecture**: Implementation of the GPT-2 model architecture with configurable parameters.
- **Distributed Data Parallel (DDP)**: Utilizes PyTorch's DistributedDataParallel for efficient multi-GPU training.
- **Mixed Precision Training**: Employs `torch.autocast` for automatic mixed precision, enhancing training speed and reducing memory usage.
- **Fused Kernel Operations**: Uses fused AdamW optimizer when available for improved performance.
- **Custom Learning Rate Scheduler**: Implements a cosine learning rate schedule with warmup.
- **Gradient Accumulation**: Supports larger effective batch sizes through gradient accumulation.
- **Efficient Data Loading**: Custom `DataLoaderLite` class for fast and memory-efficient data loading.
- **Model Compilation**: Utilizes `torch.compile()` for potential speedups (when available).
- **Weight Tying**: Implements weight tying between embedding and output layers for parameter efficiency.
- **Validation and Sampling**: Includes periodic validation loss computation and text sampling for monitoring training progress.

## Technical Details

- **Tokenization**: Uses the `tiktoken` library for GPT-2 compatible tokenization.
- **Optimizer**: AdamW with weight decay and learning rate scheduling.
- **Gradient Clipping**: Applies gradient norm clipping for stability.
- **Device Agnostic**: Supports training on CUDA, MPS (Apple Silicon), and CPU.
- **Customizable Configuration**: Easily adjustable hyperparameters through the `GPTConfig` class.

## Dataset Preparation

To download and prepare the dataset, run the following command:

```bash
python datasets/edu_fineweb10B/prepare.py
```

This script will download and process the necessary data for training.

## Usage

To run the pretraining script:

```bash
# For single GPU or CPU training
python model.py

# For multi-GPU training with DDP
torchrun --nproc_per_node=NUM_GPUS model.py
```

Replace `NUM_GPUS` with the number of GPUs you want to use.

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Acknowledgements

This project is inspired by and based on the educational content provided by Andrej Karpathy. It serves as an implementation of various advanced training techniques in the context of language model pretraining.

## License

[MIT License](LICENSE)