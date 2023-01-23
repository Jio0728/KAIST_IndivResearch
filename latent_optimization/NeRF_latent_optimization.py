"""TODO:
1. WANDB 연결
2. e4e 코드 추가: 완료
"""

import os
import sys

import numpy as np 
import matplotlib.pyplot as plt 
import os
import glob
import imageio
import torch
import torch.nn as nn
import clip
import math
import datetime

from torchvision import models 
import torch.nn.functional as F

import torch.optim as optimㅌ
import click
import dnnlib
import legacy
import copy
import PIL.Image

from collections import OrderedDict
from tqdm import tqdm
from torchvision.utils import save_image
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from tensorboardX import SummaryWriter

CUR_DIR = os.path.abspath(__file__)
TASK_DIR = os.path.dirname(CUR_DIR)
PRJ_DIR = os.path.dirname(TASK_DIR)
RESULT_DIR = os.path.join(TASK_DIR, "results")
INPUT_IMGS_DIR = os.path.join(TASK_DIR, "input_imgs")

sys.path.append(PRJ_DIR)
summary = SummaryWriter()

from StyleNeRF.apps.text_guide import CLIPLoss, IDLoss, get_lr
from StyleNeRF.training.networks import Generator
from StyleNeRF.renderer import Renderer
from modules.utils import load_yaml
from modules.optimizers import get_optimizer
from modules.features import img2latents

# ---- load config -----
config_path = f"{TASK_DIR}/config/NeRF_latent_optimization.yml"
config = load_yaml(config_path)

GPU = config['GPU']

SEED = config['SETTING']['seed']
SAVE_VIDEO = config['SETTING']['save_video']
IMG_FNAME = config['SETTING']['img_fname']

NETWORK_PKL = config['EXPERIMENT']['network_pkl']
E4E_PKL = config['EXPERIMENT']['e4e_pkl']['optimizer']
OPTIMIZER_STR = config['EXPERIMENT']
TEXT = config['EXPERIMENT']['text']
MODE = config['EXPERIMENT']['mode']
LATENT_MODE = config['EXPERIMENT']['latent_mode']
NUM_STEPS = config['EXPERIMENT']['num_steps']
LR_INIT = config['EXPERIMENT']['lr_init']
LR_RAMPUP = config['EXPERIMENT']['lr_rampup']
L2_LAMBDA = config['EXPERIMENT']['l2_lambda']
ID_LAMBDA = config['EXPERIMENT']['id_lambda']
TRUNC = config['EXPERIMENT']['trunce'] 

now = datetime.datetime.now()
formattedDate = now.strftime("%Y%m%d_%H%M%S")
SERIAL = f"latent_optimization_{formattedDate}"
OUT_DIR = os.path.join(RESULT_DIR, SERIAL)

# ----- SETTING -----
np.random.seed(SEED)
torch.manual_seed(SEED)
conv2d_gradfix.enabled = True  # Improves training speed.
os.makedirs(OUT_DIR)
device = torch.device(f"cuda:{GPU}")

if SAVE_VIDEO:
    video = imageio.get_writer(f'{OUT_DIR}/edit.mp4', mode='I', fps=24, codec='libx264', bitrate='16M')
    print (f'Saving optimization progress video "{OUT_DIR}/edit.mp4"')

# ----- Load Networks -----
if os.path.isdir(NETWORK_PKL):
    network_pkl = sorted(glob.glob(NETWORK_PKL + '/*.pkl'))[-1]
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

with torch.no_grad():
    G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
    misc.copy_params_and_buffers(G, G2, require_all=False)
G2 = Renderer(G2, None, program=None)

# ----- Obtain Latent Vector -----
if LATENT_MODE == "random":
    z = np.random.RandomState(SEED).randn(1, G.z_dim)
    ws_init = G.mapping(torch.from_numpy(z).to(device), None, truncation_psi=TRUNC) 
elif LATENT_MODE == "from_image":
    img_path = os.path.join(INPUT_IMGS_DIR, IMG_FNAME)
    assert os.path.isfile(img_path), print("latent_optimization/input_imgs에 해당하는 이미지파일이 없습니다.")
    ws_init = img2latents(img_path, E4E_PKL, device)
ws = ws_init.clone()
ws.requires_grad = True #! CHECKED

# ----- Create Initial Image -----
camera_matrices = G2.get_camera_traj(0, 1, device=device)
initial_image = G2(styles=ws_init, camera_matrices=camera_matrices)

# ----- Set Loss, Optimizer -----
optimizer = get_optimizer(OPTIMIZER_STR)
optimizer   = optimizer([ws], lr=LR_INIT, betas=(0.9,0.999), eps=1e-8)
pbar        = tqdm(range(NUM_STEPS))
text_input  = torch.cat([clip.tokenize(TEXT)]).to(device)

clip_loss   = CLIPLoss(stylegan_size=G.img_resolution)
if ID_LAMBDA > 0:
    id_loss = IDLoss()

# ----- TRAIN -----
for i in pbar:
    # t = i / float(num_steps)
    # lr = get_lr(t, lr_init, rampup=lr_rampup)
    # optimizer.param_groups[0]["lr"] = lr
    optimizer.zero_grad()

    img_gen = G2(styles=ws, camera_matrices=camera_matrices)
    c_loss = clip_loss(img_gen, text_input)

    if ID_LAMBDA > 0:
        i_loss = id_loss(img_gen, initial_image)[0]
    else:
        i_loss = 0

    if MODE == "edit":
        l2_loss = ((ws - ws_init) ** 2).sum()
        loss = c_loss + L2_LAMBDA * l2_loss + ID_LAMBDA * i_loss
    else:
        l2_loss = 0
        loss = c_loss

    loss.backward()
    optimizer.step()
    pbar.set_description((f"loss: {loss.item():.4f}; c:{c_loss.item():.4f}; l2:{l2_loss:.4f}; id:{i_loss:.4f}"))
    if i % 10 == 0:
        if SAVE_VIDEO:
            image = torch.cat([initial_image, img_gen], -1) * 0.5 + 0.5
            image = image.permute(0, 2, 3, 1) * 255.
            image = image.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(image)
            # loss_list.append(loss) #loss 추가.
            summary.add_scalar("total_loss", loss.item(), i) # TODO: WANDB로 수정
            summary.add_scalar("c_loss", c_loss.item(), i)
        
    if i % 100 == 0:
        save_image(torch.cat([initial_image, img_gen], -1).clamp(-1,1), f"{OUT_DIR}/{i}.png", normalize=True, range=(-1, 1))
        # np.save("latent_W/{}.npy".format(name),dlatent.detach().cpu().numpy())
    
# # render the learned model
# if len(kwargs) > 0:  # stylenerf
#     assert save_video
#     G2.program = 'rotation_camera3'
#     all_images = G2(styles=ws)
#     def proc_img(img): 
#         return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

#     initial_image = proc_img(initial_image * 2 - 1).numpy()[0]
#     all_images = torch.stack([proc_img(i) for i in all_images], dim=-1).numpy()[0]
#     for i in range(all_images.shape[-1]):
#         video.append_data(np.concatenate([initial_image, all_images[..., i]], 1))
    
if SAVE_VIDEO:
    video.close()