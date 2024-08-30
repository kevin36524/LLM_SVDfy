# LLM SVDfy

This repository contains code for replacing attention and MLP layers in Large Language Models (LLMs) with their SVD equivalent U and V linear layers. The goal is to explore the impact of this substitution on model performance and efficiency.

## Files

* **LLM_SVDfy.ipynb:** This Jupyter Notebook provides the implementation for replacing attention and MLP layers with their SVD equivalent U and V linear layers. 
* **train_mistral.py:** This script uses the SFTTrainer from the `trl` library to train the linear layers (U and V).
* **trainingLogs.txt:** This file contains logs from the training process, including error messages encountered during training.

## Training Environment

* **CUDA:** We are utilizing CUDA for GPU acceleration.
* **Device:** `cuda`
* **GPU:** NVIDIA A100-SXM4-40GB
* **GPU Memory:** 42.41 GB

## Encountered Issues

During training on a GCP VM with an A100 GPU, we encountered a `torch.cuda.OutOfMemoryError`.
Use code with caution.
Markdown
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU
This error indicates that the GPU memory is insufficient for the training process. Potential solutions include:

* **Reducing batch size:** Smaller batches require less memory.
* **Gradient accumulation:** Accumulate gradients over multiple steps before updating model parameters.
* **Mixed precision training:** Use FP16 or BF16 data types to reduce memory consumption.
* **Model parallelism:** Distribute the model across multiple GPUs.

## Future Work

* Investigate and implement solutions to address the `OutOfMemoryError`.
* Evaluate the performance of the SVD-based LLM against the original model.
* Explore different SVD approximation techniques to balance accuracy and efficiency.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
