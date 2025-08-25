# Recommended docker setup
```
docker run --gpus all -it -e NVIDIA_DRIVER_CAPABILITIES=all --name my_simplerenv_container    -v /NAS:/NAS     -v /SSD:/SSD     pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
```
If NVIDIA_DRIVER_CAPABILITIES=all option is missing, docker may not found proper graphic driver.

# Requirement

```
apt-get update && apt-get install -y ffmpeg vulkan-tools vulkan-validationlayers libvulkan1 libglvnd-dev
```


# Install
## Simpler Env Install

Create an conda env
```
conda create -n simpler_env python=3.10
conda activate simpler_env
conda install -c conda-forge ffmpeg=4.2.2
```

Install numpy<2.0 (otherwise, requirement dependency crash will occur in pinocchio)
```
pip install numpy==1.24.4
```

Install ManiSkill2
```
cd {this_repo}/SimplerEnv/ManiSkill2_real2sim
pip install -e .
```

Install SimplerEnv
```
cd {this_repo}/SimplerEnv
pip install -e .
```

Install other requirements
```
cd {this_repo}/SimplerEnv
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support
pip install typing-extensions==4.14.0
pip install git+https://github.com/nathanrooy/simulated-annealing
```

## OpenVLA-OFT Install
```
# Create and activate conda environment
conda create -n openvla_oft python=3.10 -y
conda activate openvla_oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
# You may skip here, if you start with 'Recommended Docker' setup.
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
cd {this_repo}/openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

# Get Started

### Deploy model server

First, download OFT model(trained on bridge v2)
```
huggingface-cli download shylee/bridge_oft2
```

Deploy OpenVLA-OFT server
```
conda activate openvla_oft
cd {this_repo}
OFT_MODEL_PATH={your model path} bash deploy_openvla.sh
```

### run benchmark

open new bash, and run the simplerenv scripts
```
conda activate simpler_env
cd {this_repo}/SimplerEnv
bash scripts/oft_bridge.sh
```

# Troubleshooting

If you encounter issues such as
```
RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.
Segmentation fault (core dumped)
```
Follow [this link](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to troubleshoot the issue. (Even though the doc points to SAPIEN 3 and ManiSkill3, the troubleshooting section still applies to the current environments that use SAPIEN 2.2 and ManiSkill2).

# Additional Tips

If you want to serve new type of model,
modifying [main_inference.py](https://github.com/marchmelo0923/SimplerEnv-OpenvlaOFT/blob/main/SimplerEnv/simpler_env/main_inference.py), and add new policy to [simpler_env/policies](https://github.com/marchmelo0923/SimplerEnv-OpenvlaOFT/tree/main/SimplerEnv/simpler_env/policies) repository.

# Citation

Thanks to [Simpler Env](https://github.com/simpler-env/SimplerEnv) and [OpenVLA OFT](https://github.com/moojink/openvla-oft)