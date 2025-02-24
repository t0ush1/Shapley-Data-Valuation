import math
import os
import pickle
import numpy as np

from baseline import IPSS_SHAP, LOO_SHAP, TMC_SHAP, GTB_SHAP, CC_SHAP
from util import Record, l2norm, load_info

expre_dir = "/code/Shapley-Data-Valuation/sources/scalability/expres"

algorithms = {"IPSS": IPSS_SHAP, "DIG-FL": LOO_SHAP, "Extended-TMC": TMC_SHAP, "Extended-GTB": GTB_SHAP, "CC-Shapley": CC_SHAP}
model = "linear_model"
cnum = 100
dataset = "emnist"
setup = "same"

if __name__ == "__main__":
    record = Record(model=model, cnum=cnum, dataset=dataset, setup=setup)
    # gamma = int(cnum * math.log2(cnum))
    gamma = 300
    results = {}
    counter = {}

    info = load_info(cnum, dataset, setup)
    zeros = np.zeros(info["zero_cnum"])
    zero_clients = info["zero_clients"]
    orig_clients = info["orig_clients"]
    same_clients = info["same_clients"]

    for alg, func in algorithms.items():
        if alg != "Extended-GTB":
            continue
        record.clear_counter()
        sv, time = func(record, cnum, gamma, False)
        counter[alg] = record.get_counter()

        shapley = np.array(sv)
        zero_err = l2norm(zeros, shapley[zero_clients])
        same_err = l2norm(shapley[orig_clients], shapley[same_clients])
        results[alg] = {"time": time, "zero_err": zero_err, "same_err": same_err, "sv": sv}

        print(alg, counter[alg], time, zero_err, same_err)
    
    print(results)
    print(counter)

    with open(os.path.join(expre_dir, f"{model}_{cnum}_{dataset}_{setup}.res"), "wb") as f:
        pickle.dump(results, f)

        
