FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages torch>=2.0 numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    torch \
    numpy \
    pandas

RUN mkdir -p /workspace/data /workspace/diagnosis /workspace/solution

COPY data/patient_records.csv /workspace/data/patient_records.csv
COPY diagnosis/current_model.py /workspace/diagnosis/current_model.py
COPY diagnosis/train_config.py /workspace/diagnosis/train_config.py

WORKDIR /workspace