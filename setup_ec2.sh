#!/usr/bin/env bash
# Bootstrap an EC2 GPU box (Deep Learning OSS Nvidia Driver AMI, Ubuntu 22.04,
# x86_64) for the Public Speaking pipeline. Idempotent — safe to re-run.
#
#   ssh ubuntu@<ip>
#   git clone <repo> && cd personalised-lms-service && bash setup_ec2.sh
#   python run_pipeline.py "<youtube-url-or-/path/to/videos>"
set -euo pipefail

echo "== system deps =="
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg git

echo "== python deps =="
python -m pip install -q -U pip
python -m pip install -q -r requirements.txt
# extras the pipeline needs beyond requirements.txt
python -m pip install -q -U yt-dlp lap openai-whisper librosa mediapipe==0.10.14

echo "== GPU check =="
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

# LLM: with a GPU, run_pipeline loads Qwen directly. Without one, install Ollama:
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "== no GPU -> installing Ollama for the LLM =="
  curl -fsSL https://ollama.com/install.sh | sh
  (ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 3
  ollama pull mistral
fi

echo "== done. run:  python run_pipeline.py <url-or-dir> =="
