#!/bin/bash

# Install Tesseract OCR
sudo apt install tesseract-ocr

# Install Python packages
pip install diffusers transformers scipy ftfy accelerate pytesseract numpy -qq

# Install Detectron2 from GitHub
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'