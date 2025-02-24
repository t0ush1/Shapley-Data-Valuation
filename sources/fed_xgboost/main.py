import json
import time
import numpy as np
import tensorflow as tf
import xgboost as xgb
import pickle as pk


def _get_xgboost_model_attr(xgb_model):
    num_parallel_tree = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"])
    num_trees = int(xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
    return num_parallel_tree, num_trees


def update_model(prev_model, model_update):
    if not prev_model:
        return model_update
    else:
        # Append all trees
        # get the parameters
        pre_num_parallel_tree, pre_num_trees = _get_xgboost_model_attr(prev_model)
        cur_num_parallel_tree, cur_num_trees = _get_xgboost_model_attr(model_update)

        # check num_parallel_tree, should be consistent
        if cur_num_parallel_tree != pre_num_parallel_tree:
            raise ValueError(f"add_num_parallel_tree should not change, previous {pre_num_parallel_tree}, current {cur_num_parallel_tree}")
        prev_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(pre_num_trees + cur_num_trees)
        # append the new trees
        append_info = model_update["learner"]["gradient_booster"]["model"]["trees"]
        for tree_ct in range(cur_num_trees):
            append_info[tree_ct]["id"] = pre_num_trees + tree_ct
            prev_model["learner"]["gradient_booster"]["model"]["trees"].append(append_info[tree_ct])
            prev_model["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
            prev_model["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(pre_num_trees + tree_ct + 1)
        # append iteration_indptr
        return prev_model


cnum = 6
data_path = f"/home/hetianran/code/TKDE-SHAP-main/sources/datasets/adult/client_{cnum}_same/"

dtrain = []
boost_round = []
for cid in range(cnum):
    fx = open(data_path + f"client_trainX_{cid}.pk", "rb")
    fy = open(data_path + f"client_trainY_{cid}.pk", "rb")
    X_train = pk.load(fx)
    y_train = pk.load(fy)
    boost_round.append(y_train.size // 200)
    dtrain.append(xgb.DMatrix(X_train, label=y_train))

fx = open(data_path + "testX.pk", "rb")
fy = open(data_path + "testY.pk", "rb")
X_test = pk.load(fx)
y_test = pk.load(fy)
dtest = xgb.DMatrix(X_test)

params = {
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


results = {}
for number in range(1 << cnum):
    subset = [i for i in range(cnum) if ((number >> i) & 1) == 1]
    print(f"Now subset is {subset}")

    global_round = 4
    global_bst = None
    global_bst_as_dict = None
    if number == 0:
        results[0] = {"loss": 1, "acc": 0, "time": 0}
        continue

    begin_time = time.perf_counter()
    for round in range(global_round):
        for cid in subset:
            local_bst = xgb.train(params, dtrain[cid], num_boost_round=boost_round[cid], xgb_model=global_bst)
            new_bst = local_bst[local_bst.num_boosted_rounds() - boost_round[cid] :]
            update = json.loads(new_bst.save_raw("json"))
            global_bst_as_dict = update_model(global_bst_as_dict, update)
        global_bst = xgb.Booster(params=params, model_file=bytearray(json.dumps(global_bst_as_dict), "utf-8"))
        
        y_pred_proba = global_bst.predict(dtest)
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)
        correct = np.sum(y_pred == y_test)
        accuracy = correct / len(y_test)

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        loss = loss_fn(y_test, y_pred_proba).numpy()

        print(f"  Round: {round} BR: {global_bst.num_boosted_rounds()} Accuracy: {accuracy:.4f} Loss: {loss:.4f}")
    end_time = time.perf_counter()

    results[number] = {"loss": loss, "acc": accuracy, "time": end_time - begin_time}

print(results)
with open(f"xgboost_model_{cnum}_adult_same.rec", "wb") as fout:
    pk.dump(results, fout)