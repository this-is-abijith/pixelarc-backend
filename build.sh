#!/usr/bin/env bash
set -e

pip install "setuptools<82" wheel
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    flask flask-cors Pillow numpy gunicorn torch torchvision
pip install "huggingface_hub==0.25.2"
pip install --no-build-isolation git+https://github.com/sberbank-ai/Real-ESRGAN.git