import json
import os
import pickle
import threading

import sys

import tqdm

sys.path.append("/code/Shapley-Data-Valuation/sources")

from helper_shap import power2number
from fed_models import linear_model

work_dir = "/code/Shapley-Data-Valuation/sources"
record_dir = os.path.join(work_dir, "rec_fed_sample_time")
data_dir = os.path.join(work_dir, "datasets")
os.chdir(work_dir)


def l2norm(list1, list2):
    l2_norm_difference = sum((x - y) ** 2 for x, y in zip(list1, list2)) ** 0.5
    l2_norm_reference = sum(y**2 for y in list1) ** 0.5
    return l2_norm_difference / l2_norm_reference if l2_norm_reference != 0 else l2_norm_difference


def load_info(cnum, dataset, setup):
    filepath = os.path.join(data_dir, dataset, f"client_{cnum}_{setup}", "info.json")
    with open(filepath, "r") as f:
        info = json.load(f)
    return info


def run_script(os_system_command, exit_codes):
    exit_codes[os_system_command] = os.system(os_system_command)


def fed_train(sample, model, cnum, dataset, setup):
    local_round = 2
    global_round = 4

    print(f"### fed_train: model={model}, cnum={cnum}, dataset={dataset}, setup={setup}, rec_sample={sample}")

    fed_client_com = [
        [
            f"CUDA_VISIBLE_DEVICES={(i + 1) % 4}",
            "python",
            "fed_mp_client.py",
            f"--model={model}",
            f"--dataset={dataset}",
            f"--client_num={cnum}",
            f"--local_round={local_round}",
            f"--rec_sample='{sample}'",
            f"--cid={cid}",
            f"--train_setup={setup}",
        ]
        for i, cid in enumerate(sample)
    ]

    fed_server_com = [
        "python",
        "fed_mp_server.py",
        f"--model={model}",
        f"--dataset={dataset}",
        f"--client_num={cnum}",
        f"--rec_sample='{sample}'",
        f"--fed_round={global_round}",
        f"--train_setup={setup}",
    ]

    os_command = []
    for com in fed_client_com:
        os_command.append(" ".join(com))
    os_command.append(" ".join(fed_server_com))

    for cid in (sample):
    # for cid in tqdm.tqdm(sample, desc="create model"):
        model = linear_model((28, 28), 10, 1)
        model.model_save(cid)
    print(f"all {len(sample)} models created")

    exit_codes = {}
    threads = [threading.Thread(target=run_script, args=(osc, exit_codes), daemon=True) for osc in os_command]
    for thd in threads:
        thd.start()
    for thd in threads:
        thd.join()
    for osc in os_command:
        assert exit_codes[osc] == 0


class Record:
    def __init__(self, model, cnum, dataset, setup):
        self.model = model
        self.cnum = cnum
        self.dataset = dataset
        self.setup = setup
        self.filepath = os.path.join(record_dir, f"{model}_{cnum}_{dataset}_{setup}.rec")
        self.record = {}
        if not os.path.exists(self.filepath):
            with open(self.filepath, "wb") as f:
                pickle.dump({}, f)
        self.load()
        self.visited = set()

    def load(self):
        with open(self.filepath, "rb") as f:
            self.record = pickle.load(f)

    def get(self, sample, type, train=True):
        index = power2number(sample)
        if index not in self.visited:
            self.visited.add(index)
        if str(index) not in self.record:
            if not train:
                return None
            fed_train(sample, self.model, self.cnum, self.dataset, self.setup)
            self.load()
        return self.record[str(index)][type]
    
    def clear_counter(self):
        self.visited = set()

    def get_counter(self):
        return len(self.visited)

# cnum = 100
# fed_train([0], "linear_model", 100, "emnist", "same")
# fed_train(list(range(100)), "linear_model", cnum, "emnist", "same")
