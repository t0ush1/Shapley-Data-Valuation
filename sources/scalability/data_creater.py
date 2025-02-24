import json
import pickle
import random
import shutil
import subprocess
import os
import numpy as np


work_dir = "/code/Shapley-Data-Valuation/sources/datasets"

if __name__ == "__main__":
    dataset = "emnist"
    cnum = 100

    zero_cnum = int(cnum * 0.05)
    same_cnum = int(cnum * 0.05)
    norm_cnum = cnum - zero_cnum - same_cnum

    command = [
        "python",
        "fed_data_creater.py",
        f"--dataset={dataset}",
        f"--c_num={norm_cnum}",
        "--c_type=same",
    ]
    subprocess.run(command, cwd=work_dir)

    norm_dir = os.path.join(work_dir, dataset, f"client_{norm_cnum}_same")
    data_dir = os.path.join(work_dir, dataset, f"client_{cnum}_same")
    os.rename(norm_dir, data_dir)

    zero_clients = list(range(norm_cnum, norm_cnum + zero_cnum))
    for cid in zero_clients:
        with open(os.path.join(data_dir, f"client_trainX_{cid}.pk"), "wb") as fout:
            pickle.dump(np.array([]), fout)
        with open(os.path.join(data_dir, f"client_trainY_{cid}.pk"), "wb") as fout:
            pickle.dump(np.array([]), fout)

    orig_clients = sorted(random.sample(range(norm_cnum), same_cnum))
    same_clients = list(range(cnum - same_cnum, cnum))
    for i, cid in enumerate(same_clients):
        src = os.path.join(data_dir, f"client_trainX_{orig_clients[i]}.pk")
        dst = os.path.join(data_dir, f"client_trainX_{cid}.pk")
        shutil.copy(src, dst)
        src = os.path.join(data_dir, f"client_trainY_{orig_clients[i]}.pk")
        dst = os.path.join(data_dir, f"client_trainY_{cid}.pk")
        shutil.copy(src, dst)

    info_path = os.path.join(data_dir, "info.json")
    info = {
        "norm_cnum": norm_cnum,
        "zero_cnum": zero_cnum,
        "same_cnum": same_cnum,
        "zero_clients": zero_clients,
        "orig_clients": orig_clients,
        "same_clients": same_clients,
    }
    with open(info_path, "w") as fout:
        json.dump(info, fout, indent=4)
    print(info)
