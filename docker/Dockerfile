# SuperMimic runtime: Isaac Lab 3 (main) kit-less image + tmux/ffmpeg/compiler.
FROM isaac-lab-kitless:main-46d6cc3
USER root
RUN apt-get update && apt-get install -y --no-install-recommends tmux ffmpeg util-linux build-essential python3-dev && rm -rf /var/lib/apt/lists/*
# Register the NVIDIA EGL vendor (the driver library is injected by the NVIDIA container runtime);
# without it GLVND only knows Mesa and MuJoCo's offscreen renderer falls back to CPU rasterization.
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
