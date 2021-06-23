import os
import numpy as np
from disc.samples import get_1a_sample, get_1b_sample, get_1c_sample, get_1d_sample, get_2a_sample, get_2b_sample, get_2c_sample, get_2d_sample, get_2b_sample_simple, get_2c_sample_simple, get_2d_sample_simple
from disc.utils import save_image
from tqdm import tqdm

levels = {"1a": get_1a_sample, 
            "1b": get_1b_sample,
            "1c": get_1c_sample,
            "1d": get_1d_sample,
            "2a": get_2a_sample,
            "2b": get_2b_sample,
            "2c": get_2c_sample,
            "2d": get_2d_sample,
            "2b_s": get_2b_sample_simple,
            "2c_s": get_2c_sample_simple,
            "2d_s": get_2d_sample_simple}

def create_dataset(n, level, c, out_dir, h=128, w=128, seed=1234):
    f_sample = levels[level]

    class_dir = os.path.join(out_dir, str(c))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    for i in tqdm(range(n)):
        sample = f_sample(c, h, w)
        sample_path = os.path.join(class_dir, f"{i}.png")
        save_image(sample, sample_path, norm=True)

if __name__ == "__main__":
    create_dataset(5000, "1a", 0, "datasets/1a")
