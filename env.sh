pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 23.08
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
