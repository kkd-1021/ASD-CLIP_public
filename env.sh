pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 23.08
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
# to avoid lapjv version conflict
pip uninstall lapjv numpy
pip install --upgrade numpy
pip install lapjv --no-cache-dir

# to avoid numba version conflict
pip uninstall numba llvmlite
pip install --no-cache-dir numba llvmlite