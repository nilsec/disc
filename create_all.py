from fade.datasets import create_dataset
import os

level_to_classes = {"1a": [0,1,2],
                    "1b": [0,1,2],
                    "1c": [0,1,2],
                    "1d": [0,1],
                    "2a": [0,1],
                    "2b": [0,1],
                    "2c": [0,1],
                    "2d": [0,1],
                    "2b_s": [0,1],
                    "2c_s": [0,1],
                    "2d_s": [0,1]}

def create_all(base_dir="/nrs/funke/ecksteinn/soma_data"):
    n = 5000
    seed = 1234
    for level, classes in level_to_classes.items():
        print(level)
        if not level in ["2d_s"]:
            continue
        out_dir = os.path.join(base_dir, level)

        for c in classes:
            create_dataset(n, level, c, out_dir, 128, 128, seed=seed)

if __name__ == "__main__":
    create_all()
   
