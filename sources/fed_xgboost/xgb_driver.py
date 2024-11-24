import pickle
import threading

from xgb_server import run_server
from xgb_client import run_client

cnum = 10
DATAPATH = f"../datasets/adult/client_{cnum}_same/"
PARAMS = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "lambda": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eta": 0.01,
    "nthread": 4
}

if __name__ == "__main__":
    results = {"0": {"loss": 1, "acc": 0, "time": 0}}
    for index in range(1, 1 << cnum):
        subset = [i for i in range(cnum) if (index >> i) & 1]
        print(f"Now subset is {subset}")
        threads = [threading.Thread(target=run_client, args=(cid, DATAPATH, PARAMS), daemon=True) for cid in subset]
        threads.append(threading.Thread(target=run_server, args=(index, subset, results, DATAPATH, PARAMS), daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    print(results)
    with open(f"xgboost_model_{cnum}_adult_same.rec", "wb") as fout:
        pickle.dump(results, fout)
        