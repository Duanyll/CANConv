# Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening

*Yule Duan, Xiao Wu, Haoyu Deng, Liang-Jian Deng*

[![Static Badge](https://img.shields.io/badge/CVPR_2024-Accepted-green)](https://openaccess.thecvf.com/content/CVPR2024/html/Duan_Content-Adaptive_Non-Local_Convolution_for_Remote_Sensing_Pansharpening_CVPR_2024_paper.html) [![Static Badge](https://img.shields.io/badge/arXiv-2404.07543-brown?logo=arxiv)
](https://arxiv.org/abs/2404.07543)

![The overall workflow of CANConv.](imgs/main.png)

Abstract: Currently, machine learning-based methods for remote sensing pansharpening have progressed rapidly. However, existing pansharpening methods often do not fully exploit differentiating regional information in non-local spaces, thereby limiting the effectiveness of the methods and resulting in redundant learning parameters. In this paper, we introduce a so-called content-adaptive non-local convolution (CANConv), a novel method tailored for remote sensing image pansharpening. Specifically, CANConv employs adaptive convolution, ensuring spatial adaptability, and incorporates non-local self-similarity through the similarity relationship partition (SRP) and the partition-wise adaptive convolution (PWAC) sub-modules. Furthermore, we also propose a corresponding network architecture, called CANNet, which mainly utilizes the multi-scale self-similarity. Extensive experiments demonstrate the superior performance of CANConv, compared with recent promising fusion methods. Besides, we substantiate the method's effectiveness through visualization, ablation experiments, and comparison with existing methods on multiple test sets. The source code is publicly available.

## Getting Started

### Environment Setup with Docker

**Please prepare a Docker environment with CUDA support:**

- Ensure you have Docker installed on your system.
- To enable CUDA support within the Docker environment, refer to the official Docker documentation for setting up GPU acceleration: Docker GPU setup: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

If you cannot use Docker, you can also set up the environment manually. However, you may run into issues with the dependencies.

1. **Clone the repo and its submodules:**
    
   ```bash
   git clone --recurse-submodules https://github.com/duanyll/CANConv.git
   ```

2. **Edit mount point for datasets in `.devcontainer/devcontainer.json`:**
    - Locate the `.devcontainer/devcontainer.json` file within the cloned repo.
    - Specify the path to your datasets on your host machine by adjusting the `mounts` configuration in the file.

3. **Reopen the repo in VS Code devcontainer:**
    - Open the cloned repo in VS Code.
    - When prompted, select "Reopen in Container" to activate the devcontainer environment.
    - It may take serval minutes when pulling the base PyTorch image and install requirements for the first time.

4. **Install pacakges and build native libraries**
   - If you are using the devcontainer, you can skip this step, vscode will automatically run the script.
   
   ```bash
   bash ./build.sh
   ```

5. **Train the model:**
    
   ```bash
   python -m canconv.scripts.train cannet wv3
   ```

   - Replace `cannet` with other networks available in the `canconv/models` directory.
   - Replace `wv3` with other datasets defined in `presets.json`.
   - Results are placed in the `runs` folder.

## Additional Information

**Pretrained weights:**
- Pre-trained weights can be found in the `weights` folder.

**Datasets:**
- Datasets are used from the repo [liangjiandeng/PanCollection](https://github.com/liangjiandeng/PanCollection).

**Metrics:**
- Metrics are obtained using tools from [liangjiandeng/DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) (specifically, the `02-Test-toolbox-for-traditional-and-DL(Matlab)` directory).

## Known Issues

- The code is not adapted for using multiple GPUs. If you have multiple GPUs, you can only utilize one GPU for training.
   - If you have to use a device other than `cuda:0`, you have to use `CUDA_VISIBLE_DEVICES` to specify the GPU device.
   - For example, to use the second GPU, you can run `CUDA_VISIBLE_DEVICES=1 python -m canconv.scripts.train cannet wv3`.
   - Notes: Though the Python code respects the `device` option in the configuration file, the C++ code contains direct calls to cuBLAS functions, which may not respect the device option. The `CUDA_VISIBLE_DEVICES` environment variable is the most reliable way to specify the GPU device.