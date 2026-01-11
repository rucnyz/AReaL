FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/sensors \
    && mkdir -p /var/log/sensors \
    && mkdir -p /opt/archive \
    && mkdir -p /tmp/analytics \
    && mkdir -p /root

COPY data/input.dat /opt/sensors/input.dat
COPY pipeline.sh /opt/sensors/pipeline.sh
COPY data/sensor_data.log.2023-01-15 /opt/archive/sensor_data.log.2023-01-15
COPY data/sensor_data.log.2023-02-10 /opt/archive/sensor_data.log.2023-02-10
COPY data/sensor_data.log.2023-03-22 /opt/archive/sensor_data.log.2023-03-22
COPY data/sensor_data.log.2023-04-18 /opt/archive/sensor_data.log.2023-04-18
COPY data/sensor_data.log.2023-05-25 /opt/archive/sensor_data.log.2023-05-25
COPY data/sensor_data.log.2023-06-30 /opt/archive/sensor_data.log.2023-06-30

RUN chmod +x /opt/sensors/pipeline.sh

WORKDIR /root