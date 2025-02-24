# get the file name to pickle dump
import os 
import pickle as pk
from copy import deepcopy
import time
import numpy as np


def get_rec_file_name(model_name, client_num, dataset_name, train_setup='same'):
    return model_name + "_" + str(client_num) +"_" + dataset_name + "_" + train_setup +".rec"


# from utils.shapley_value import power2number buildPowerSets
def power2number(itemset):
    if type(itemset) == str:
        itemset = eval(itemset)
    number = 0
    for i in itemset:
        number+= 1<<i
    return number 


def buildPowerSets(itemSet):
    number = 0
    if type(itemSet)==list:
        number = len(itemSet)
    elif type(itemSet)==int:
        number = itemSet
    allSet = list(range(number))
    pwSet = [[] for i in range (1<<number)]
    for i in range(1<<number):
        # subSet = []
        for j in range(number):
            if (i>>j)%2 == 1:
                pwSet[i].append(itemSet[j])
    return pwSet


def rec_grad(pre_weights, now_weights, cid, round, now_args, datasize):
    if not os.path.exists('./rec_fed_grad/'): os.mkdir('./rec_fed_grad/')
    file_path = "./rec_fed_grad/" + get_rec_file_name(now_args.model, now_args.client_num, now_args.dataset, now_args.train_setup)[:-4] + "/"
    if not os.path.exists(file_path): os.mkdir(file_path)

    file_name = str(cid)+ "_" +str(round)+".grad_rec"

    # record the datasize of each client.
    if round==0:
        with open(file_path+ str(cid)+'.datasize','wb') as fout:
            pk.dump(datasize, fout)    

    grad = [now-pre for (pre, now) in zip(pre_weights, now_weights)]

    # save the data in pickle form 
    with open(file_path+file_name,'wb') as fout:
        pk.dump(grad, fout)
    print("SAVE:", file_path+file_name)


def load_grad(cid, round, now_args):
    file_path = "./rec_fed_grad/" + get_rec_file_name(now_args.model, now_args.client_num, now_args.dataset, now_args.train_setup)[:-4] + "/"
    file_name = str(cid)+ "_" +str(round)+".grad_rec"
    return pk.load(file_path+file_name)

def rec_sample_time(args, accuarcy, loss, time_slot, output_flag=True): 
    if not os.path.exists("./rec_fed_sample_time/"): os.mkdir("./rec_fed_sample_time/")
    file_name_rec_time = ''
    if args.now_gpu == -1:
        file_name_rec_time = "./rec_fed_sample_time/" + get_rec_file_name(args.model, args.client_num, args.dataset, args.train_setup)
    else:
        file_name_rec_time = "./rec_fed_sample_time/" + get_rec_file_name(args.model, args.client_num, args.dataset, args.train_setup)
        file_name_rec_time = file_name_rec_time[:-4] + '_'+ str(args.now_gpu)+file_name_rec_time[-4:]
    if not os.path.exists(file_name_rec_time):
        print("now we create the " + file_name_rec_time)
        with open(file_name_rec_time, 'wb') as fout: 
            blank_dict = {}
            pk.dump(blank_dict, fout)
    
    with open(file_name_rec_time, "rb") as fin:
        now_rec = pk.load(fin)
    now_rec[str(power2number(args.rec_sample))] = {'loss':loss, 'acc': accuarcy, 'time': time_slot}
    if output_flag:
        print(now_rec)
    with open(file_name_rec_time, 'wb') as fout:
        pk.dump(now_rec, fout)


def handle_sample_time(output_flag=True): 

    now_rec = {}
    now_rec['0'] = {'loss':0, 'acc': 0, 'time': 0}
    now_rec['1'] = {'loss':0, 'acc': 100, 'time': 0} # c1
    now_rec['2'] = {'loss':0, 'acc': 125, 'time': 0} # c2 
    now_rec['3'] = {'loss':0, 'acc': 270, 'time': 0} # c1, c2
    now_rec['4'] = {'loss':0, 'acc': 50, 'time': 0}  # c3
    now_rec['5'] = {'loss':0, 'acc': 375, 'time': 0} # c1, c3
    now_rec['6'] = {'loss':0, 'acc': 350, 'time': 0} # c2, c3
    now_rec['7'] = {'loss':0, 'acc': 500, 'time': 0} # c1, c2, c3
    
    return now_rec

# def rec_OR_sample_time():
def exist_rec_tmr_results(args_model, args_client_num, args_dataset, args_train_setup):
    if not os.path.exists("./rec_fed_tmr_res/"): os.mkdir("./rec_fed_tmr_res/")   
    file_name_rec_time = "./rec_fed_tmr_res/" + 'tmr_' + args_model + '_' + str(args_client_num) + '_' + args_dataset + '_'+ args_train_setup + '.rec'
    if not os.path.exists(file_name_rec_time): 
        return False 
    return True


def exist_rec_sample_results(args_model, args_client_num, args_dataset, args_train_setup, args_rec_sample, agrs_now_gpu=-1):
    file_name_rec_time = ''
    if agrs_now_gpu == -1:
        file_name_rec_time = "./rec_fed_sample_time/" + get_rec_file_name(args_model, args_client_num, args_dataset, args_train_setup)
    else:
        file_name_rec_time = "./rec_fed_sample_time/" + get_rec_file_name(args_model, args_client_num, args_dataset, args_train_setup)
        file_name_rec_time = file_name_rec_time[:-4] + '_'+ str(agrs_now_gpu)+file_name_rec_time[-4:]
    
    if not os.path.exists("./rec_fed_sample_time/"): os.mkdir("./rec_fed_sample_time/")
    # file_name_rec_time = "./rec_fed_sample_time/" + get_rec_file_name(args_model, args_client_num, args_dataset, args_train_setup)
    
    if not os.path.exists(file_name_rec_time):
        print("now we create the " + file_name_rec_time)
        with open(file_name_rec_time, 'wb') as fout: 
            blank_dict = {}
            pk.dump(blank_dict, fout)

    with open(file_name_rec_time, "rb") as fin:
        now_rec = pk.load(fin)
    if now_rec.get(str(power2number(args_rec_sample))) == None: return False
    
    return True


def load_rec_sample_results(args_model, args_client_num, args_dataset, args_train_setup):
    file_name_rec_time = "./rec_fed_sample_time/" + get_rec_file_name(args_model, args_client_num, args_dataset, args_train_setup)
    with open(file_name_rec_time, "rb") as fin:
        now_rec = pk.load(fin)
    # if now_rec.get(str(power2number(args_rec_sample))) == None: return False
    return now_rec


import os
import pickle as pk
import numpy as np
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_trunc_multi_round(client_num, now_round, fed_round, args):
     responses, global_model, grad_alpha, test_dataset, args_model, args_dataset, args_train_setup = args
     testX, testY = test_dataset

     all_sets = [i for i in range(client_num)]
     power_set = buildPowerSets(all_sets)
     # rec_res = [[] for _ in range(fed_round)]
     all_alpha = sum(grad_alpha)

     if not os.path.exists("./rec_fed_tmr_res/"): os.mkdir("./rec_fed_tmr_res/")   
     file_name_tmr = "./rec_fed_tmr_res/" + 'tmr_' + args_model + '_' + str(client_num) + '_' + args_dataset + '_'+ args_train_setup + '.rec'
    
     if not os.path.exists(file_name_tmr):
         with open(file_name_tmr, 'wb') as fout: 
             blank_rec = [{} for _ in range(fed_round)]
             pk.dump(blank_rec, fout)

     global_model_weights = global_model.model_get_weights()
     # compute round acc/loss/time of each rounds
     for subset in power_set:

         begin_time = time.perf_counter()
         temp = [np.zeros(gmw_layer.shape) for gmw_layer in global_model_weights]
         sub_alpha = sum([grad_alpha[_] for _ in subset])

         for layer in range(len(global_model_weights)):
             for cid in subset:
                 temp[layer] = temp[layer] + grad_alpha[cid]*responses[cid][layer]*all_alpha/sub_alpha
              
         sub_model_weights = deepcopy(temp)
         global_model.model_load_weights(sub_model_weights)
         test_loss, test_acc = global_model.model_get_eval(testX, testY, notes=str(now_round))

         end_time = time.perf_counter()

         # save the tmr res of rounds and combinations
         with open(file_name_tmr, "rb") as fin:
             now_rec = pk.load(fin)     
         now_rec[now_round][str(power2number(subset))] = {'loss':test_loss, 'acc': test_acc, 'time': end_time-begin_time}
         with open(file_name_tmr, 'wb') as fout:
             pk.dump(now_rec, fout)      

                
