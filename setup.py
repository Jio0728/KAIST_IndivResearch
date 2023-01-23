import os
import sys

PRJ_DIR = os.path.dirname(__file__)
sys.path.append(PRJ_DIR)

os.system('pip install ftfy regex tqdm gdown')
os.system('pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html')
os.system('pip install Ninja')
print("install requirements.txt start")
os.system('pip install -r requirements.txt')
print("install requirements.txt finished")