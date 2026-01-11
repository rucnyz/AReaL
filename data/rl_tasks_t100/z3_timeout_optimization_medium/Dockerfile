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
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages z3-solver pycparser

RUN mkdir -p /home/user/symbolic_executor /home/user/config

COPY symbolic_executor/simple.c /home/user/symbolic_executor/simple.c
COPY symbolic_executor/nested.c /home/user/symbolic_executor/nested.c
COPY symbolic_executor/loops.c /home/user/symbolic_executor/loops.c
COPY config/config.yaml /home/user/config/config.yaml

WORKDIR /home/user