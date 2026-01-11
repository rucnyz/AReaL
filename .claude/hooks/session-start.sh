#!/bin/bash

if [ -n "$CLAUDE_ENV_FILE" ]; then
  echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> "$CLAUDE_ENV_FILE"
  echo 'conda activate areal' >> "$CLAUDE_ENV_FILE"
fi

exit 0
