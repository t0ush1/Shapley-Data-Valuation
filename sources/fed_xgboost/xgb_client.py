from concurrent import futures
import json
import pickle
import threading
import xgboost as xgb
import grpc

import xgb_proto_pb2_grpc
import xgb_proto_pb2

class BoosterServicer(xgb_proto_pb2_grpc.BoosterServiceServicer):
    def __init__(self, cid, dtrain, boost_round, params):
        self.cid = cid
        self.dtrain = dtrain
        self.boost_round = boost_round
        self.params = params

    def get_booster(self, request, context):
        global_bst = xgb.Booster(params=self.params, model_file=bytearray(request.raw_booster)) if request.raw_booster != b'{}' else None
        local_bst = xgb.train(self.params, self.dtrain, self.boost_round, xgb_model=global_bst)
        new_bst = local_bst[local_bst.num_boosted_rounds() - self.boost_round :]
        update = new_bst.save_raw("json")
        return xgb_proto_pb2.bst_reply(raw_booster=bytes(update))


class StopServicer(xgb_proto_pb2_grpc.StopServiceServicer):
    def __init__(self, cid, stop_event):
        self.cid = cid
        self.stop_event = stop_event

    def stop(self, request, context):
        self.stop_event.set()
        return xgb_proto_pb2.stop_reply()


def run_client(cid, datapath, params):
    fx = open(datapath + f"client_trainX_{cid}.pk", "rb")
    fy = open(datapath + f"client_trainY_{cid}.pk", "rb")
    X_train = pickle.load(fx)
    y_train = pickle.load(fy)
    boost_round = y_train.size // 200
    dtrain = xgb.DMatrix(X_train, label=y_train)

    stop_event = threading.Event()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    xgb_proto_pb2_grpc.add_BoosterServiceServicer_to_server(BoosterServicer(cid, dtrain, boost_round, params), server)
    xgb_proto_pb2_grpc.add_StopServiceServicer_to_server(StopServicer(cid, stop_event), server)
    server.add_insecure_port(f"[::]:{50075 + cid}")

    server.start()
    stop_event.wait()
