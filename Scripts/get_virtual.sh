python3 -m venv ecg_venv
source ecg_venv/bin/activate
pip install wfdb
pip install numpy
pip install torch
pip install torchvision
pip install pandas

bash Scripts/get_datasets.sh