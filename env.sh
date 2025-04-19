pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 23.08
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
