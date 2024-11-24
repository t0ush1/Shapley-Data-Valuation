import argparse as ap
import os
import pickle as pk
import json
from PIL import Image
import numpy as np

DATASET = "celeba"
IMAGE_SIZE = 84
IMAGE_DIR = "/code/leaf/data/celeba/data/raw/img_align_celeba"
TEST_PATH = "/code/leaf/data/celeba/data/test/all_data_iid_01_1_keep_0_test_9.json"
TRAIN_PATH = "/code/leaf/data/celeba/data/train/all_data_iid_01_1_keep_0_train_9.json"
DATASETS_DIR = "/code/TKDE-SHAP-main/sources/datasets"


def load_img(img_names):
    imgs = []
    for name in img_names:
        with Image.open(f"{IMAGE_DIR}/{name}") as img:
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
            imgs.append(np.array(img))
    return imgs


def create_dataset(cnum, setup):
    output_dir = f"{DATASETS_DIR}/{DATASET}/client_{cnum}_{setup}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(TRAIN_PATH, "r") as fin:
        train_json = json.load(fin)
    train_x, train_y = [], []
    for user in train_json["users"]:
        train_x.extend(train_json["user_data"][user]["x"])
        train_y.extend(train_json["user_data"][user]["y"])
    datasize = len(train_x) // cnum
    client_train_x = [train_x[datasize * cid : datasize * (cid + 1)] for cid in range(cnum)]
    client_train_y = [train_y[datasize * cid : datasize * (cid + 1)] for cid in range(cnum)]
    client_train_img = [load_img(x) for x in client_train_x]
    for cid in range(cnum):
        with open(f"{output_dir}/client_trainX_{cid}.pk", "wb") as fout:
            pk.dump(np.array(client_train_img[cid]), fout)
        with open(f"{output_dir}/client_trainY_{cid}.pk", "wb") as fout:
            pk.dump(np.array(client_train_y[cid]), fout)
    
    with open(TEST_PATH, "r") as fin:
        test_json = json.load(fin)
    test_x, test_y = [], []
    for user in test_json["users"]:
        test_x.extend(test_json["user_data"][user]["x"])
        test_y.extend(test_json["user_data"][user]["y"])
    test_img = load_img(test_x)
    with open(f"{output_dir}/testX.pk", "wb") as fout:
        pk.dump(np.array(test_img), fout)
    with open(f"{output_dir}/testY.pk", "wb") as fout:
        pk.dump(np.array(test_y), fout)



if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Create Celeba Federated Dataset.")
    parser.add_argument("--cnum", type=int, default=3)
    parser.add_argument("--setup", type=str, default="same")
    args = parser.parse_args()
    create_dataset(args.cnum, args.setup)
