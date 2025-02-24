import logging
import fed_proto_pb2
import fed_proto_pb2_grpc
import grpc
import time
from copy import deepcopy
from helper_shap import create_trunc_multi_round

import os
import argparse as ap  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from fed_models import rpcio_to_nparray, nparray_to_rpcio
from fed_models import *
from helper_shap import *
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D as mp3d

import numpy as np
import pickle as pk
import multiprocessing as mproc
import threading
from queue import Queue


# # for device in gpu_devices:
# tf.config.experimental.set_virtual_device_configuration(gpu_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=200)])


def get_weights_from_client(grad_stub, grad_data, grad_type, grad_shape, cname, share_q):
        print("Now we call the local train from client#{}".format(cname))

        local_weights = grad_stub.grad_descent(fed_proto_pb2.server_request(
            server_grad_para_data=grad_data,
            server_grad_para_type=grad_type, 
            server_grad_para_shape=grad_shape))
        
        print("Now we put local_weights in share_queue of {}.".format(cname))
        share_q.put(local_weights)
        # return 
        # return local_weights
        

# class call_grad_descent_from_client(fed_proto_pb2_grpc.GradDescentServiceServicer):
# here the fed_server calls the remote function of clients.
def run_fed_server(_args, _basic_port=BASIC_PORT):

    # load the parameters of FL
    c_num = _args.client_num
    now_gpu = _args.now_gpu
    _sample_client = eval(_args.rec_sample)
    fed_model = FED_MODEL_DICT[_args.model]
    fed_round = _args.fed_round
    print("Now the combination of client is ", _sample_client)

    # build the channel to each client to execute the local trainning.
    ports = [c+_basic_port+ now_gpu*PORT_GAP for c in range(c_num)]  
    channels= []
    for cid in range(c_num):
        if cid in _sample_client:
            channels.append(grpc.insecure_channel("localhost:"+str(ports[cid]), options=[
                ('grpc.max_send_message_length', 10 * 1024 * 1024),
                ('grpc.max_receive_message_length', 10 * 1024 * 1024)
            ]))
    grad_stubs = [fed_proto_pb2_grpc.GradDescentServiceStub(channel) for channel in channels]
    size_stubs = [fed_proto_pb2_grpc.GetDataSizeServiceStub(channel) for channel in channels]
    stop_stubs = [fed_proto_pb2_grpc.stop_serverStub(channel) for channel in channels]


    print("# build the channel to each client to execute the local trainning.")

    # Loaded the test data
    print("# Loaded the test data")
    data_path = "./datasets/"+_args.dataset+"/client_"+str(c_num)+"_"+ _args.train_setup +"/"
    file_testX = open(data_path+'testX.pk', 'rb')
    file_testY = open(data_path+'testY.pk', 'rb')
    testX = pk.load(file_testX)
    testY = pk.load(file_testY)

    # Globally train the federated model, i.e. execute the stub.grad_decendent
    print("# Globally train the federated model, i.e. execute the stub.grad_decendent.")
    data_shape = FED_SHAPE_DICT[_args.dataset]
    global_model = fed_model(data_shape[0], data_shape[1])

    # Check whether the subset of client is []
    if len(_sample_client)==0:
        test_loss, test_acc = global_model.model_get_eval(testX, testY)
        # stop_stus = fed_proto_pb2_grpc.stop_serverStub(grpc.insecure_channel("localhost:"+str(STOP_PORT)))
        # stop_stus.stop(fed_proto_pb2.stop_request(message="simplex"))
        return test_acc, test_loss        

    print("Next execute FL trainning!")
    global_model_weights = global_model.model_get_weights()
    datasize_response = [size_stubs[c].get_datasize(fed_proto_pb2.datasize_request(size=0)) for c in range(len(size_stubs))]
    np_datasize = np.array([r.size for r in datasize_response])
    all_datasize = np.sum(np_datasize)

    print(" ## The datasize from rpc is :", np_datasize, " ## allsize is %d"%(all_datasize))
    grad_alpha = (np_datasize/all_datasize)

    # share_queue = [mproc.Queue() for _ in range(len(_sample_client))]
    share_queue = [Queue() for _ in range(len(_sample_client))]

    for now_round in range(fed_round):
        grad_data, grad_type, grad_shape = nparray_to_rpcio(global_model_weights) # data, type, shape

        rpcio_responses = []
        for t in range((len(_sample_client) - 1) // 20 + 1):
            print("begin:", sum(tf.config.experimental.get_memory_usage(f'GPU:{gpu}') for gpu in range(4)))

            threadings = []
            group = list(range(t * 20, min((t + 1) * 20, len(_sample_client))))
            for c in group:
                thead =  threading.Thread(target=get_weights_from_client, 
                                        args=(grad_stubs[c], grad_data, grad_type, grad_shape, _sample_client[c], share_queue[c]),
                                        daemon=True,
                                        name="client_"+str(c))
                print("Start the Proc-{}".format(_sample_client[c]))
                thead.start()
                threadings.append(thead)
            for thd in threadings:
                thd.join()
            print("All threadings joined.")
            for c in group: assert share_queue[c].empty() == False
            rpcio_responses += [share_queue[c].get() for c in group]

            print("end:", sum(tf.config.experimental.get_memory_usage(f'GPU:{gpu}') for gpu in range(4)))
        
        print("Have got the response of weights from client.")
        responses = [rpcio_to_nparray(r.client_grad_para_data, r.client_grad_para_type, r.client_grad_para_shape) for r in rpcio_responses]

        temp = [np.zeros(gmw_layer.shape) for gmw_layer in global_model_weights]
        for layers in range(len(global_model_weights)):
            # for i in range(len(responses)):
            for cid in range(len(responses)):
                # exec aggregation on each layer
                temp[layers] = temp[layers] + grad_alpha[cid]*responses[cid][layers]

        if args.tmr_flag == 1:
            create_trunc_multi_round(len(responses), now_round, fed_round, (responses, global_model, grad_alpha, (testX, testY), _args.model, _args.dataset, _args.train_setup))
        
        global_model_weights = deepcopy(temp)
        # test the performance of global model.
        global_model.model_load_weights(global_model_weights)
        test_loss, test_acc = global_model.model_get_eval(testX, testY)
        print("Subset<-{}: Round#{}#, Acc:{}, Loss:{}".format(_sample_client, now_round, test_acc, test_loss))

        
        # 
    
    # stop_stus = fed_proto_pb2_grpc.stop_serverStub(grpc.insecure_channel("localhost:"+str(STOP_PORT)))
    for stop_req in stop_stubs:
        # print("Now we request to stop the")
        stop_req.stop(fed_proto_pb2.stop_request(message="simplex"))
    
    # record the loss and acc
    test_loss, test_acc = global_model.model_get_eval(testX, testY)
    return test_acc, test_loss

if __name__ == "__main__":
    logging.basicConfig()

    # load parameter of run_fed_server
    parser = ap.ArgumentParser(description="Creating Info. for Comp. Shapley.")
    parser.add_argument("--model", type=str, default='linear')
    parser.add_argument("--client_num", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="emnist")
    parser.add_argument("--rec_sample", type=str, default=str([i for i in range(5)]))
    parser.add_argument("--fed_round", type=int, default=5)
    parser.add_argument("--tmr_flag", type=int, default=0)
    parser.add_argument("--train_setup", type=str, default='same')
    parser.add_argument("--now_gpu", type=int, default=-1)

    

    args = parser.parse_args()

    # waiting for the build-up of client
    seconds = max(5, len(eval(args.rec_sample)))
    for i in range(seconds):
        print("The Server is waiting for the client until %d seconds."% (seconds-i))
        time.sleep(1)

    # run the server in FL & record the running time of FL
    begin_time = time.perf_counter()
    acc, loss = run_fed_server(args)
    end_time = time.perf_counter()


    # record the time of this sample in FL
    if args.tmr_flag == 0:
        rec_sample_time(args, acc, loss, end_time-begin_time)

