#!/bin/bash

# Sensor Data Processing Pipeline Recovery Script

set -e

# Step 1: Check the current state and identify issues
echo "Starting pipeline recovery..."

# Step 2: Clean up any artifacts from failed runs
files_removed=0
space_freed_bytes=0

# Remove lock files if they exist
for lockfile in /opt/sensors/*.lock /var/log/sensors/*.lock /opt/archive/*.lock /tmp/analytics/*.lock; do
    if [ -f "$lockfile" ]; then
        size=$(stat -f%z "$lockfile" 2>/dev/null || stat -c%s "$lockfile" 2>/dev/null || echo 0)
        rm -f "$lockfile"
        files_removed=$((files_removed + 1))
        space_freed_bytes=$((space_freed_bytes + size))
    fi
done

# Remove partial or temporary files
for tmpfile in /var/log/sensors/*.tmp /opt/archive/*.tmp /tmp/analytics/*.tmp; do
    if [ -f "$tmpfile" ]; then
        size=$(stat -f%z "$tmpfile" 2>/dev/null || stat -c%s "$tmpfile" 2>/dev/null || echo 0)
        rm -f "$tmpfile"
        files_removed=$((files_removed + 1))
        space_freed_bytes=$((space_freed_bytes + size))
    fi
done

# Step 3: Clean up old backup files in /opt/archive/ to free space
# Remove dated backup files (sensor_data.log.YYYY-MM-DD format)
if [ -d "/opt/archive/" ]; then
    for backup_file in /opt/archive/sensor_data.log.[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]; do
        if [ -f "$backup_file" ]; then
            size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null || echo 0)
            rm -f "$backup_file"
            files_removed=$((files_removed + 1))
            space_freed_bytes=$((space_freed_bytes + size))
        fi
    done
fi

# Convert bytes to MB (rounded down)
space_freed_mb=$((space_freed_bytes / 1024 / 1024))

# Step 4: Ensure all necessary directories exist with proper permissions
mkdir -p /var/log/sensors
mkdir -p /opt/archive
mkdir -p /tmp/analytics

# Step 5: Verify the pipeline script exists and is executable
if [ ! -f "/opt/sensors/pipeline.sh" ]; then
    echo "Error: Pipeline script not found at /opt/sensors/pipeline.sh"
    exit 1
fi

chmod +x /opt/sensors/pipeline.sh

# Step 6: Test if the pipeline can run successfully
pipeline_ready=false

if /opt/sensors/pipeline.sh > /dev/null 2>&1; then
    pipeline_ready=true
    echo "Pipeline test successful"
else
    echo "Pipeline test failed"
fi

# Step 7: Verify /opt/archive has at least 50MB free space
archive_free_space=$(df -m /opt/archive | awk 'NR==2 {print $4}')
if [ "$archive_free_space" -lt 50 ]; then
    pipeline_ready=false
    echo "Warning: /opt/archive has less than 50MB free space"
fi

# Step 8: Create the solution.json file
cat > /root/solution.json <<EOF
{
  "files_removed": $files_removed,
  "space_freed_mb": $space_freed_mb,
  "pipeline_ready": $pipeline_ready
}
EOF

echo "Recovery complete. Solution written to /root/solution.json"
exit 0