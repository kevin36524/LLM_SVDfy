LLM_SVDfy.ipynb ==> Used to replace all the attention layers and the mlp layers with SVD equivalent U an V linear layers 
train_mistral.py ==> Using SFTTrainer from trl to train the linear layers.
trainingLogs.txt ==> I am getting error when training on GCP VM with A100 gpu. 

Note we are using cuda
Using device: cuda
GPU Name: NVIDIA A100-SXM4-40GB
Total GPU Memory: 42.41 GB

torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU 
