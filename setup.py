import os
import sys

PRJ_DIR = os.path.dirname(__file__)
print(PRJ_DIR)
sys.path.append(PRJ_DIR)

os.system('pip install -r requirements.txt')
os.system('pip install ftfy regex tqdm gdown')