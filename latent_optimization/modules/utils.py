import yaml
import math

def load_yaml(yml_path):
    with open(yml_path) as f:
        yml_file = yaml.load(f, Loader=yaml.FullLoader)
        print(yml_file)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp