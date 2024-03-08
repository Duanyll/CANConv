# Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening

![The overall workflow of CANConv.](https://cdn.duanyll.com/img/20240308152348.png)

Here we provide the source code for the CVPR 2024 paper **Content-Adaptive Non-Local Convolution for Remote Sensing Pansharpening**. 

Abstract: Currently, machine learning-based methods for remote sensing pansharpening have progressed rapidly. However, existing pansharpening methods often do not fully exploit differentiating regional information in non-local spaces, thereby limiting the effectiveness of the methods and resulting in redundant learning parameters. In this paper, we introduce a so-called content-adaptive non-local convolution (CANConv), a novel method tailored for remote sensing image pansharpening. Specifically, CANConv employs adaptive convolution, ensuring spatial adaptability, and incorporates non-local self-similarity through the similarity relationship partition (SRP) and the partition-wise adaptive convolution (PWAC) sub-modules. Furthermore, we also propose a corresponding network architecture, called CANNet, which mainly utilizes the multi-scale self-similarity. Extensive experiments demonstrate the superior performance of CANConv, compared with recent promising fusion methods. Besides, we substantiate the method's effectiveness through visualization, ablation experiments, and comparison with existing methods on multiple test sets. The source code is publicly available.



## Getting Started

### Environment Setup

**Please prepare a Docker environment with CUDA support:**

- Ensure you have Docker installed on your system.
- To enable CUDA support within the Docker environment, refer to the official Docker documentation for setting up GPU acceleration: Docker GPU setup: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Using the Repo

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

4. **Build native libraries:**
    
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
