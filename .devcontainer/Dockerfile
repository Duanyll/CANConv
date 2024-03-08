FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y git gdb vim curl wget tmux zip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com h5py einops tqdm matplotlib tensorboard torch-tb-profiler ninja scipy