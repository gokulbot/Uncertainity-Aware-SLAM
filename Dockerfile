# ------------------------------------------------------------
# Base image: CUDA 12.4 + cuDNN on Ubuntu 22.04
# ------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Basic utilities and Python
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y unzip sudo git wget python3-pip ffmpeg \
    libsm6 libxext6 libgtk-3-dev libxkbcommon-x11-0 vulkan-tools \
    software-properties-common locales curl gnupg lsb-release gosu && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Add universe/multiverse repos
# ------------------------------------------------------------
RUN add-apt-repository universe && add-apt-repository multiverse && apt-get update

# ------------------------------------------------------------
# Locale setup
# ------------------------------------------------------------
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ------------------------------------------------------------
# Upgrade pip + install Python packages
# ------------------------------------------------------------
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
      "pypose>=0.6.8" opencv-python-headless evo \
      matplotlib tabulate tqdm rich cupy-cuda12x einops \
      "timm==0.9.12" "rerun-sdk==0.21.0" yacs numpy \
      pyyaml wandb pillow scipy flow_vis h5py \
      "xformers==0.0.27.post2" onnx torchvision \
      jaxtyping "typeguard==2.13.3"

# ------------------------------------------------------------
# ===== ROS 2 Humble =====
# ------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y ros-humble-desktop && \
    echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# ------------------------------------------------------------
# ===== OpenCV 4.x (CUDA) =====
# ------------------------------------------------------------
WORKDIR /opt
RUN apt-get update && apt-get install -y \
      build-essential cmake git pkg-config \
      libjpeg-dev libpng-dev libtiff-dev \
      libavcodec-dev libavformat-dev libswscale-dev \
      libv4l-dev libxvidcore-dev libx264-dev \
      libgtk-3-dev libcanberra-gtk-module libcanberra-gtk3-module \
      libtbb-dev libopenexr-dev libatlas-base-dev gfortran python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --branch 4.x --depth 1 https://github.com/opencv/opencv.git && \
    git clone --branch 4.x --depth 1 https://github.com/opencv/opencv_contrib.git && \
    mkdir -p /opt/opencv/build && cd /opt/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D WITH_CUDNN=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 \
          -D BUILD_opencv_python_bindings_generator=OFF \
          -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_DOCS=OFF \
          -D CUDA_ARCH_BIN="7.5 8.6" .. && \
    make -j$(nproc --ignore=2 || echo 4) && make install && ldconfig

# ------------------------------------------------------------
# ===== g2o =====
# ------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/RainerKuemmerle/g2o.git && \
    mkdir -p g2o/build && cd g2o/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && make install && ldconfig

# ------------------------------------------------------------
# ===== GTSAM =====
# ------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/borglab/gtsam.git && \
    cd gtsam && git checkout 4.2 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
          -DGTSAM_BUILD_TESTS=OFF \
          -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
          -DGTSAM_USE_SYSTEM_EIGEN=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && make install && ldconfig

# ------------------------------------------------------------
# ===== LibTorch =====
# ------------------------------------------------------------
WORKDIR /opt
RUN curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip && \
    unzip libtorch.zip && rm libtorch.zip

ENV torch_DIR=/opt/libtorch/share/cmake/Torch
ENV CMAKE_PREFIX_PATH=${torch_DIR}
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/libtorch/lib:$LD_LIBRARY_PATH

# ------------------------------------------------------------
# Entrypoint (ROOT always)
# ------------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

