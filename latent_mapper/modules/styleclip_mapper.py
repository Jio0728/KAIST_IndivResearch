import torch
from torch import nn
import sys
import os 
import copy
import numpy as np 
import matplotlib.pyplot as plt 
import os
import glob
import imageio
import math
CUR_PATH = os.path.abspath(__file__)
TASK_DIR = os.path.dirname(os.path.dirname(CUR_PATH))
PRJ_DIR = os.path.dirname(TASK_DIR)
StyleNeRF_DIR = os.path.join(TASK_DIR, "StyleNeRF")
StyleCLIP_DIR = os.path.join(TASK_DIR, "StyleCLIP")
sys.path.append(PRJ_DIR)
# sys.path.append("./StyleNeRF")

# for i in range(len(sys.path)):
#   print(sys.path[i])

# print("is training folder accessible?:", os.path.isdir("./StyleNeRF/training"))



from StyleCLIP.mapper import latent_mappers
# from models.stylegan2.model import Generator
from StyleNeRF.training import networks
from StyleNeRF.torch_utils import misc
# from StyleNeRF import legacy
# import legacy as legacy
import StyleNeRF.legacy as legacy
# from renderer import Renderer
from StyleNeRF.renderer import Renderer
import StyleNeRF.dnnlib as dnnlib
# import dnnlib as dnnlib
import glob


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class StyleCLIPMapper(nn.Module):

	def __init__(self, opts):
		super(StyleCLIPMapper, self).__init__()
		self.opts = opts
		# Define architecture
		self.mapper = self.set_mapper()
		# self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        # ------ 여기부터 ------
		if os.path.isdir(self.opts.styleNeRFckpt):
			network_pkl = sorted(glob.glob(self.opts.styleNeRFckpt + '/*.pkl'))[-1]
		print('Loading networks from "%s"...' % network_pkl)
		
		with dnnlib.util.open_url(network_pkl) as fp: 
			# print("is generator in sys.modules?:", training.networks in sys.modules)      
			G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
		G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

		with torch.no_grad():
			G2 = networks.Generator(*G.init_args, **G.init_kwargs).to(device)
			misc.copy_params_and_buffers(G, G2, require_all=False)
		G2 = Renderer(G2, None, program=None)
	
        # ----- 여기까지 styleNeRF generator 받아오기. -----
		self.decoder = G2
		self.olddecoder = G
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights() #styleNeRF 쓸 때는 self.decoder할 때 이미 pretrained 가져와서 load_weights 안씀.

	def set_mapper(self):
		if self.opts.work_in_stylespace:
			mapper = latent_mappers.WithoutToRGBStyleSpaceMapper(self.opts)
		elif self.opts.mapper_type == 'SingleMapper':
			mapper = latent_mappers.SingleMapper(self.opts)
		elif self.opts.mapper_type == 'LevelsMapper':
			mapper = latent_mappers.LevelsMapper(self.opts)
		else:
			raise Exception('{} is not a valid mapper'.format(self.opts.mapper_type))
		return mapper

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)
			# self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
		# else: # styeCLIP은 styleGAN decoder를 사용하고, styleGAN pretrained weights를 불러오는 코드. 근데 난 NeRF generator를 디코더로 쓰니까 없앰.
		# 	print('Loading decoder weights from pretrained!')
		# 	ckpt = torch.load(self.opts.stylegan_weights)
		# 	self.decoder.load_state_dict(ckpt['g_ema'], strict=False)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.mapper(x)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images