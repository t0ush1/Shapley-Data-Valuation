import logging
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
from concurrent import futures
# import os
import argparse as ap  
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fed_models import rpcio_to_nparray, nparray_to_rpcio
from fed_models import *
from helper_shap import *

import threading
import numpy as np
import pickle as pk
# import tensorflow as tf
import time

# Parameters for Fed_Client
SERVE_STOP_FLAG = False 
OUPUT_INFO = True


# for device in gpu_devices:
    # tf.config.experimental.set_virtual_device_configuration(device,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=200)])
    # tf.config.experimental.set_memory_growth(device, True)
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpu_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=200)])

# for device in gpu_devices:
    # tf.config.experimental.set_memory_growth(device, True)
    # configTensorFlow.gpu_options.per_process_gpu_memory_fraction = memoryLimited / memoryTotal


class datasize_servicier(fed_proto_pb2_grpc.GetDataSizeServiceServicer):
    def __init__(self, _dataset, _cid):
        self.datasize = len(_dataset[1])
        self.cid = _cid

    def get_datasize(self, request, context):
        return fed_proto_pb2.datasize_reply(size=self.datasize)

class grad_descent_servicer(fed_proto_pb2_grpc.GradDescentServiceServicer):
    def __init__(self, _model, _dataset, _cid, _args):
        self.dataset = _dataset
        data_shape = FED_SHAPE_DICT[_args.dataset]
        self.serial = len(eval(_args.rec_sample)) > 20
        self.model = _model(data_shape[0], data_shape[1])
        self.cid = _cid
        self.local_round = _args.local_round
        self.rec_grad = _args.rec_grad
        # print("arg.rec_grad={}, self.rec_grad={}".format(_args.rec_grad, self.rec_grad))
        self.local_batch = 16 if _args.dataset != "sent140" else 64
        self.global_round = 0 # record the global round we are in.
        self.args = _args

    def grad_descent(self, request, context):

        # Download the global model by grpc (rpcio-->np.array)
        print("The client %d executes the local trainning with %d epoches" %(self.cid, self.local_round))
        byte_data = list(request.server_grad_para_data)
        byte_shape = list(request.server_grad_para_shape)
        byte_type = list(request.server_grad_para_type)
        global_model_weights = rpcio_to_nparray(byte_data, byte_type, byte_shape)

        # if self.serial or self.model.model == None:
        #     self.model.model_load(self.cid)
        self.model.model_load_weights(global_model_weights)
        
        # Locally train the globally model
        self.model.model_fit(self.dataset, self.local_round, self.local_batch)
        client_model_weights = self.model.model_get_weights()
        
        # Record the grad of this client
        # print("Now rec_grad is {}".format(self.rec_grad))
        if self.rec_grad==1:
            rec_grad(global_model_weights, client_model_weights, self.cid, self.global_round, self.args, len(self.dataset[1]))
        self.global_round += 1

        # reply the updates model by grpc (np.array-->rpcio)
        byte_data, byte_type, byte_shape = nparray_to_rpcio(client_model_weights)
        print("Reply Gradients to Server by client %d" %(self.cid))

        # if self.serial:
        #     self.model.model_save(self.cid)

        return fed_proto_pb2.client_reply(client_grad_para_data=byte_data, 
                                          client_grad_para_type=byte_type, 
                                          client_grad_para_shape=byte_shape)
    

class stop_server(fed_proto_pb2_grpc.stop_serverServicer):
    def __init__(self, _stop_event, _cid=0):
        self.stop_event = _stop_event
        self.cid = _cid

    def stop(self, request, context):
        print("The client %d received the stop request from " % (self.cid)+ request.message)
        global SERVE_STOP_FLAG
        SERVE_STOP_FLAG = True
        self.stop_event.set()
        return fed_proto_pb2.stop_reply(message="The client %d has been stopped!" % (self.cid))

def run_fed_client(_args):

    cid = _args.cid
    stop_event = threading.Event()
    now_gpu = _args.now_gpu

    # prepare the dataset for each client
    data_path = "./datasets/"+ _args.dataset + "/client_"+ str(_args.client_num) + '_'+ _args.train_setup +"/"
    file_client_trainX = open(data_path+'client_trainX_'+str(cid)+'.pk', 'rb')
    file_client_trainY = open(data_path+'client_trainY_'+str(cid)+'.pk', 'rb')
    client_trainX = pk.load(file_client_trainX)
    client_trainY = pk.load(file_client_trainY)
    # client_trainX = client_trainX[:client_trainX.shape[0]//2]
    # client_trainY = client_trainY[:client_trainY.shape[0]//2]
    client_dataset = (client_trainX, client_trainY)

    # if OUPUT_INFO:
    #     print("Loaded Trainning Dataset for #cid={} (all {}) clients, rec_grad={}.".format(cid ,_args.client_num, _args.rec_grad))
    #     print("In Client {}:  \n  DataPath:{} \n  DataName:{}".format(cid, data_path, data_path+'client_trainX_'+str(cid)+'.pk'))
    #     print("  DataSize={}".format(client_trainX.shape))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_send_message_length', 10 * 1024 * 1024),
        ('grpc.max_receive_message_length', 10 * 1024 * 1024)
    ])
    fed_proto_pb2_grpc.add_GetDataSizeServiceServicer_to_server(datasize_servicier(client_dataset, cid), server)
    fed_proto_pb2_grpc.add_GradDescentServiceServicer_to_server(grad_descent_servicer(FED_MODEL_DICT[_args.model], client_dataset, cid, _args), server)
    fed_proto_pb2_grpc.add_stop_serverServicer_to_server(stop_server(stop_event, cid), server)
    server.add_insecure_port("[::]:"+str(BASIC_PORT+PORT_GAP*now_gpu + cid))

    print("Rpc_Client_{} is created.".format(cid))
    server.start()
    stop_event.wait()
    print("The client {} have been stoped. ".format(cid))

if __name__ == "__main__":

    parser = ap.ArgumentParser(description="Creating Info. for Comp. Shapley.")
    parser.add_argument("--model", type=str, default='linear')
    parser.add_argument("--client_num", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="emnist")
    parser.add_argument("--local_round", type=int, default=5)
    parser.add_argument("--rec_grad", type=int, default=0)
    parser.add_argument("--rec_sample", type=str, default=str([i for i in range(5)]))
    parser.add_argument("--train_setup", type=str, default='same')
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--now_gpu", type=int, default=-1)

    # parser.add_argument("--tmr_flag", type=int, default=0)

    # parser.add_argument("--rec_sample", type=bool, default=False)
    args = parser.parse_args()
    # print("For client#{}, rec_grad is {}".format(args.cid, args.rec_grad))
    logging.basicConfig()
    run_fed_client(args)




