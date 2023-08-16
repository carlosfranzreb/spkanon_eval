conda activate ./venv

cd submodules
git submodule add https://github.com/yl4579/StarGANv2-VC
git mv StarGANv2-VC StarGANv2VC
cd ..

# StarGAN requirements
pip install --no-input munch==2.5.0
pip install --no-input parallel-wavegan==0.5.5

# download StarGAN weights
pip install gdown
mkdir checkpoints
cd checkpoints
mkdir stargan
cd stargan
gdown 1nzTyyl-9A1Hmqya2Q_f2bpZkUoRjbZsY
unzip Models.zip
rm Models.zip
gdown 1q8oSAzwkqi99oOGXDZyLypCiz0Qzn3Ab
unzip Vocoder.zip
rm Vocoder.zip
pip uninstall -y gdown
cd ../..