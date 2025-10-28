# ============================================================
# Base image: Ubuntu 22.04 + CUDA 12.4 + cuDNN + Dev tools
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Basic development dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl unzip pkg-config \
    software-properties-common \
    libgoogle-glog-dev libatlas-base-dev libeigen3-dev libboost-all-dev \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# ===== Install ROS 2 Humble (on Ubuntu 22.04) =====
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    apt-get install -y curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y ros-humble-desktop && \
    echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# ------------------------------------------------------------
# ===== Install OpenCV 4.9 with CUDA =====
# ------------------------------------------------------------
WORKDIR /opt
RUN git clone --branch 4.9.0 https://github.com/opencv/opencv.git && \
    git clone --branch 4.9.0 https://github.com/opencv/opencv_contrib.git && \
    mkdir -p /opt/opencv/build && cd /opt/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D WITH_CUDNN=ON \
          -D CUDA_ARCH_BIN="7.5 8.6 8.9" \
          -D BUILD_opencv_python_bindings_generator=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          .. && \
    make -j$(nproc) && make install && ldconfig

# ------------------------------------------------------------
# ===== Install g2o =====
# ------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/RainerKuemmerle/g2o.git && \
    mkdir -p g2o/build && cd g2o/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && make install && ldconfig

# ------------------------------------------------------------
# ===== Install GTSAM =====
# ------------------------------------------------------------
WORKDIR /opt
RUN git clone https://github.com/borglab/gtsam.git && \
    mkdir -p gtsam/build && cd gtsam/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
          -DGTSAM_BUILD_TESTS=OFF \
          -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
          -DGTSAM_USE_SYSTEM_EIGEN=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && make install && ldconfig

# ------------------------------------------------------------
# ===== Install LibTorch (C++ PyTorch 2.4 + CUDA 12.4) =====
# ------------------------------------------------------------
WORKDIR /opt
RUN wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip
ENV Torch_DIR=/opt/libtorch
ENV CMAKE_PREFIX_PATH=${Torch_DIR}

# ------------------------------------------------------------
# ===== Build MAC_VO =====
# ------------------------------------------------------------
WORKDIR /workspace/MAC_VO
COPY . /workspace/MAC_VO

RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# ------------------------------------------------------------
# ===== Environment Setup =====
# ------------------------------------------------------------
ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/libtorch/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH
SHELL ["/bin/bash", "-c"]

# ------------------------------------------------------------
# Default command
# ------------------------------------------------------------
CMD ["bash"]
