FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install SPIN model checker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    spin \
    gcc \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

# Create project directory
RUN mkdir -p /workspace/verification_project

# Copy project files to the correct locations
COPY mutex.pml /workspace/verification_project/mutex.pml
COPY verify.sh /workspace/verification_project/verify.sh
COPY README.md /workspace/verification_project/README.md

# Make verify.sh executable
RUN chmod +x /workspace/verification_project/verify.sh

# Set working directory
WORKDIR /workspace/verification_project