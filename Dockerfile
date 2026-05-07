# ----------------------------------------------------------------------
#                           IMPORTANT NOTE: 
#   IF YOU MACHINE DOES NOT HAVE AN NVIDIA GPU, YOU WILL NEED TO COMMENT
#   OUT CERTAIN PORTIONS OF THE DOCKERFILE OUTLINED BELOW. THIS WILL BUILD
#   THE CONTAINER IN A CPU SPECIFIC CONFIGURATION.
# ----------------------------------------------------------------------



# ----------------------------------------------
# LEAVE UNCOMMENTED IF MACHINE HAS NVIDIA GPU
# COMMENT OUT IF RUNNING IN CPU ONLY CONFIG 
# ----------------------------------------------
FROM nvidia/cuda:13.1.0-base-ubuntu24.04

# # ----------------------------------------------
# # LEAVE COMMENTED OUT IF MACHINE HAS NVIDIA GPU
# # UNCOMMENT IF RUNNING IN CPU ONLY CONFIG
# # ----------------------------------------------
# FROM ubuntu:24.04


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-venv \
    python3-full \
    libgl1 \
    libosmesa6 \
    libglew2.2 \
    libglfw3 \
    libxinerama1 \
    libxcursor1 \
    libxrandr2 \
    libxi6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# ----------------------------------------------
# LEAVE UNCOMMENTED IF MACHINE HAS NVIDIA GPU
# COMMENT OUT IF RUNNING IN CPU ONLY CONFIG 
# ----------------------------------------------
RUN pip install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu124

# # ----------------------------------------------
# # LEAVE COMMENTED OUT IF MACHINE HAS NVIDIA GPU
# # UNCOMMENT IF RUNNING IN CPU ONLY CONFIG
# # ----------------------------------------------
# RUN pip install --no-cache-dir --break-system-packages \
#     torch --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir --break-system-packages \
    mujoco==3.6.0 \
    "gymnasium[robotics]==1.2.3" \
    stable-baselines3==2.8.0 \
    numpy \
    imageio \
    pandas \
    matplotlib \
    tensorboard==2.20.0

# ----------------------------------------------
# LEAVE UNCOMMENTED IF MACHINE HAS NVIDIA GPU
# COMMENT OUT IF RUNNING IN CPU ONLY CONFIG 
# ----------------------------------------------
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# # ----------------------------------------------
# # LEAVE COMMENTED OUT IF MACHINE HAS NVIDIA GPU
# # UNCOMMENT IF RUNNING IN CPU ONLY CONFIG
# # ----------------------------------------------
# ENV MUJOCO_GL=osmesa
# ENV PYOPENGL_PLATFORM=osmesa

WORKDIR /workspace

ADD https://api.github.com/repos/GraysonGilbert/sim2real-furuta-pendulum/commits/main /dev/null
RUN git clone https://github.com/GraysonGilbert/sim2real-furuta-pendulum.git

CMD python -c "import mujoco; import torch; print(f'MuJoCo ready! CUDA available: {torch.cuda.is_available()}');"