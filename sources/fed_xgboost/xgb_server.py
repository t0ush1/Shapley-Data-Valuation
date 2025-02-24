import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import pickle
import threading
import time
import grpc
import numpy as np
import tensorflow as tf
import xgboost as xgb

import xgb_proto_pb2
import xgb_proto_pb2_grpc


def update_model(prev_model, model_update):
    if not prev_model:
        return model_update
    else:
        pre_num_trees = int(prev_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
        cur_num_trees = int(model_update["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
        prev_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(pre_num_trees + cur_num_trees)
        append_info = model_update["learner"]["gradient_booster"]["model"]["trees"]
        for tree_ct in range(cur_num_trees):
            append_info[tree_ct]["id"] = pre_num_trees + tree_ct
            prev_model["learner"]["gradient_booster"]["model"]["trees"].append(append_info[tree_ct])
            prev_model["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
            prev_model["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(pre_num_trees + tree_ct + 1)
        return prev_model


def get_update(idx, bst_stub, global_bst_as_dict, updates):
    raw_booster = json.dumps(global_bst_as_dict).encode()
    response = bst_stub.get_booster(xgb_proto_pb2.bst_request(raw_booster=raw_booster))
    updates[idx] = json.loads(response.raw_booster)


def run_server(index, subset, results, datapath, params):
    time.sleep(5)

    begin_time = time.perf_counter()
    
    global_round = 4
    global_bst = None
    global_bst_as_dict = {}

    channels= [grpc.insecure_channel(f"localhost:{50075 + cid}") for cid in subset]
    bst_stubs = [xgb_proto_pb2_grpc.BoosterServiceStub(channel) for channel in channels]
    stop_stubs = [xgb_proto_pb2_grpc.StopServiceStub(channel) for channel in channels]

    fx = open(datapath + "testX.pk", "rb")
    fy = open(datapath + "testY.pk", "rb")
    X_test = pickle.load(fx)
    y_test = pickle.load(fy)
    dtest = xgb.DMatrix(X_test)
    
    for round in range(global_round):
        updates = [None for _ in subset]
        
        threads = [threading.Thread(target=get_update, args=(i, stub, global_bst_as_dict, updates), daemon=True) for i, stub in enumerate(bst_stubs)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        for update in updates:
            global_bst_as_dict = update_model(global_bst_as_dict, update)
        global_bst = xgb.Booster(params=params, model_file=bytearray(json.dumps(global_bst_as_dict), "utf-8"))

        y_pred_proba = global_bst.predict(dtest)
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)
        correct = np.sum(y_pred == y_test)
        accuracy = correct / len(y_test)

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        loss = loss_fn(y_test, y_pred_proba).numpy()

        print(f"  Round: {round} BR: {global_bst.num_boosted_rounds()} Accuracy: {accuracy:.4f} Loss: {loss:.4f}")
    for stub in stop_stubs:
        stub.stop(xgb_proto_pb2.stop_request())

    end_time = time.perf_counter()
    results[str(index)] = {"loss": loss, "acc": accuracy, "time": end_time - begin_time}
