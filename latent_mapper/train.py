"""
main train을 돌리는 파일.
trainer 역할을 하는 coach.py, styleclip_mapper.py options는 latent_mapper 내부에서 받아오고,
나머지는 다 StyleCLIP 서브 모듈 내에서 받아옴.
"""
import os
import json
import sys
import pprint

CUR_PATH = os.path.abspath(__file__)
TASK_DIR = os.path.dirname(CUR_PATH)
PRJ_DIR = os.path.dirname(TASK_DIR)
sys.path.append(PRJ_DIR)

from options.train_options import TrainOptions
from modules.coach import Coach


def main(opts):
	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	opts = TrainOptions().parse()
	main(opts)
