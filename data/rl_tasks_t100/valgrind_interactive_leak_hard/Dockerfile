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
    apt-get install -y valgrind && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/bin /home/agent

COPY data_processor /usr/local/bin/data_processor
COPY data/sample_data.txt /home/agent/sample_data.txt

RUN chmod +x /usr/local/bin/data_processor

WORKDIR /home/agent