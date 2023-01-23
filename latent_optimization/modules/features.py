"""TODO:
1. generate_inversions 수정 필요."""


import torch
import numpy as np
import sys
import os

CUR_DIR = os.path.abspath(__file__)
TASK_DIR = os.path.dirname(os.path.dirname(CUR_DIR))
PRJ_DIR = os.path.dirname(TASK_DIR)
sys.append(PRJ_DIR)

from encoder4editing.scripts.inference import get_all_latents, get_latents, generate_inversions, save_image, run_alignment
from encoder4editing.configs import data_configs, paths_config

from encoder4editing.datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from encoder4editing.utils.model_utils import setup_model
# from utils.common import tensor2im
# from utils.alignment import align_face
# from PIL import Image

def img2latents(images_dir, e4e_pkl, device, batch=1, n_sample=None, align=None, save_dir=None, latents_only=True): 
    """
    save_dir: save_dir에 latent 저장하는 용도 아님. 저장된 latent로 추가 훈련하고 싶을 때 활용."""
    args = (batch, n_sample, align, images_dir)
    net, opts = setup_model(e4e_pkl, device)
    generator = net.decoder
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    # Check if latents exist
    latents_file_path = os.path.join(save_dir, 'latents.pt')
    if os.path.exists(latents_file_path):
        latent_codes = torch.load(latents_file_path).to(device)
    else:
        latent_codes = get_all_latents(net, data_loader, n_sample, is_cars=False)
        # torch.save(latent_codes, latents_file_path) #저장 목적 아님.

    if not latents_only: # inversion image 생성 모듈인데, 수정 필요함.
        generate_inversions(args, generator, latent_codes, is_cars=False)

    return latent_codes


# ----- args 사용 안 하도록 지오 수정. -----
def setup_data_loader(args, opts):
    (batch, n_sample, align, images_dir) = args
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = images_dir if images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    if align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if n_sample is None:
        n_sample = len(test_dataset)
    return args, data_loader


# ----- args 사용 안 하도록 지오 수정. -----
@torch.no_grad()
def generate_inversions(args, g, latent_codes, is_cars):
    (batch, n_sample, align, images_dir) = args
    print('Saving inversion images')
    inversions_directory_path = os.path.join(images_dir, 'inversions') # 이미지 디렉토리 내에 inversion 디렉토리 형성
    os.makedirs(inversions_directory_path, exist_ok=True)
    for i in range(min(n_sample, len(latent_codes))):
        imgs, _ = g([latent_codes[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
        save_image(imgs[0], inversions_directory_path, i + 1)
