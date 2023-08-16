# create and activate conda environment
conda create -p ./venv python=3.9 -y
conda activate ./venv

# install conda packages
conda install -c conda-forge pynini -y
conda install 'ffmpeg<4.4' -y

# install pip dependencies
pip install --no-input --upgrade pip
pip install --no-input Cython==0.29.34
pip install --no-input -r ./requirements.txt

# Add NISQA as a submodule
mkdir submodules
cd submodules
git submodule add https://github.com/gabrielmittag/NISQA.git
cd ..