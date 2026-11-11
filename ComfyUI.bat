python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-comfyui.txt