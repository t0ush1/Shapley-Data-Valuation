import os
import argparse as ap  
import time 
from helper_shap import *
import multiprocessing
import threading
import argparse as ap  
import logging



# exist_dataset_names = ["emnist", "gldv2", "stackoverflow", "cifa100", "shakespeare"]

# exist_model_names = ["cnn_model"]
# client_numbers = [15]
# global_round = [4]
# local_round = [2]
# exist_dataset_names = ["emnist"]


# exist_model_names = ["linear_model"]
# client_numbers = [3, 5, 10, 15]
# global_round = [10]
# local_round = [4]
# exist_dataset_names = ["emnist"]



# Setup for  MNIST with different settings.
# exist_model_names = ["cnn_model"]
# client_numbers = [3, 5]
# global_round = [4]
# local_round = [2]
# exist_dataset_names = ["mnist"]
# exist_setup_names = ['same', 'mixDtr', 'mixSize', 'noiseX', 'noiseY']



# exist_model_names = ["linear_model"]
# client_numbers = [3, 5]
# global_round = [10]
# local_round = [4]
# exist_dataset_names = ["mnist"]
# exist_setup_names = ['same', 'mixDtr', 'mixSize', 'noiseX', 'noiseY']




exit_codes = {}


def run_script(os_system_command, exit_codes):
    exit_codes[os_system_command] = os.system(os_system_command)

def create_sample_data_mp(dataset, model, client_num, local_round, rec_sample, fed_round, tmr_flag=0, setup='same', now_gpu=-1):

    rec_grad = 0
    if len(rec_sample) == client_num:
        rec_grad = 1

    print('###,dataset={}, model={}, cnum={}, rec_grad={}, lr={}, rec_sample={}, gr={}, setup={}, now_gpu={}'.format(dataset, model, client_num, rec_grad, local_round, rec_sample, fed_round, setup, now_gpu))

    fed_client_com =[['python fed_mp_client.py ' + 
               '--model='+ "'"+model+"'" + ' ' +
               '--dataset='+ "'"+dataset+"'"  + ' ' +
               '--client_num=' + str(client_num) + ' ' +
               '--local_round=' + str(local_round) + ' ' +
               '--rec_sample=' + "'"+str(rec_sample)+"'" + ' ' +
               '--rec_grad=' + str(rec_grad) + ' '
               '--cid='+ str(cid)+ ' '
               '--train_setup='+str(setup) + ' '
               '--now_gpu='+str(now_gpu) + ' '] for cid in range(client_num)]
    
    fed_server_com = ['python fed_mp_server.py ' + 
                    '--model='+ "'"+model+"'" + ' ' +
                    '--dataset='+ "'"+dataset+"'"  + ' ' +
                    '--client_num=' + str(client_num) + ' ' +
                    '--rec_sample=' + "'"+str(rec_sample)+"'" + ' ' +
                    '--fed_round=' + str(fed_round) + ' ' +
                    '--tmr_flag=' + str(tmr_flag) + ' '
                    '--train_setup='+str(setup) + ' '
                    '--now_gpu='+str(now_gpu) + ' '] 
      
    threads = []
    os_command = []
    for i, cid in enumerate(rec_sample):
        os_command.append(f"CUDA_VISIBLE_DEVICES={(i + 1) % 2} " + fed_client_com[cid][0])
    
    os_command.append(fed_server_com[0])

    for osc in os_command: print(osc)

    threads = [threading.Thread(target=run_script, args=(osc, exit_codes), daemon=True) for osc in os_command]
    for thd in threads:
        thd.start()
    for thd in threads:
        thd.join()
    for osc in os_command: assert exit_codes[osc] == 0

if __name__ == "__main__":
    # if nam
    TEST_MODE = False
    logging.basicConfig()
    # load parameter of run_fed_Driver
    parser = ap.ArgumentParser(description="Creating Info. for all Setups in Fed_Driver.")
    parser.add_argument("--model", type=str, default='cnn_model')
    parser.add_argument("--client_num", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="emnist")
    parser.add_argument("--local_round", type=int, default=2)
    parser.add_argument("--global_round", type=int, default=4)
    parser.add_argument("--train_setup", type=str, default='same')
    parser.add_argument("--all_gpu", type=int, default=-1)
    parser.add_argument("--now_gpu", type=int, default=-1)
    args = parser.parse_args()
    
    
    exist_model_names = [args.model]
    client_numbers = [args.client_num]
    global_round = [args.global_round]
    local_round = [args.local_round]
    exist_dataset_names = [args.dataset]
    exist_setup_names = [args.train_setup]

    if TEST_MODE:
        local_round = [2]
        global_round= [4]
        CNUM = 3
        rec_sample = [0]
        # rec_sample = [i for i in range(CNUM)]

        # create_sample_data_mp(dataset='emnist', model="cnn_model", client_num=CNUM, 
        #                 local_round=local_round[0], rec_sample=[i for i in range(CNUM)], fed_round=global_round[0])
        create_sample_data_mp(dataset='adult', model="linear_model", client_num=CNUM, 
                        local_round=local_round[0], rec_sample=rec_sample, fed_round=global_round[0], tmr_flag=0, setup='same')
    # # [i for i in range(CNUM)]
    elif args.all_gpu == -1:
        # if ""
        for dataset in exist_dataset_names:
            for setup in exist_setup_names:
                for model in exist_model_names:
                    for c_num in client_numbers:
                        for g_round in global_round:
                            for l_round in local_round:
                                all_set = [i for i in range(c_num)]
                                power_set = buildPowerSets(all_set)

                                
                                # create results for trunced-multi-round methods, requiring 'all_set'.
                                if not exist_rec_tmr_results(model, c_num, dataset, setup):    
                                    create_sample_data_mp(dataset=dataset, model=model, client_num=c_num, 
                                                    local_round=l_round, rec_sample=all_set, fed_round=g_round, tmr_flag=1, setup=setup)  
                                
                                
                                # create results for other sampling-based methods
                                for subset in power_set:
                                    # print("Now subset is {}".format(subset))
                                    print("Now #Subset is {}".format(power2number(subset)))
                                    if not exist_rec_sample_results(model, c_num, dataset, setup, subset):
                                        create_sample_data_mp(dataset=dataset, model=model, client_num=c_num, 
                                                    local_round=l_round, rec_sample=subset, fed_round=g_round, tmr_flag=0, setup=setup)
                                        

    else:
        for dataset in exist_dataset_names:
            for setup in exist_setup_names:
                for model in exist_model_names:
                    for c_num in client_numbers:
                        for g_round in global_round:
                            for l_round in local_round:
                                all_set = [i for i in range(c_num)]
                                power_set = buildPowerSets(all_set)

                                all_gpu = args.all_gpu
                                now_gpu  = args.now_gpu
                                max_set_number = power2number(all_set)+1
                                # create results for other sampling-based methods
                                for subset in power_set:
                                    if max_set_number*(now_gpu-1)/all_gpu <= power2number(subset) and power2number(subset) < max_set_number*now_gpu/all_gpu:
                                        # print("Now subset is {}".format(subset))
                                        print("Now #Subset is {}".format(power2number(subset)))
                                        if not exist_rec_sample_results(model, c_num, dataset, setup, subset, now_gpu):
                                            create_sample_data_mp(dataset=dataset, model=model, client_num=c_num, 
                                                        local_round=l_round, rec_sample=subset, fed_round=g_round, tmr_flag=0, setup=setup, now_gpu=now_gpu)
                                        
                                # create results for trunced-multi-round methods, requiring 'all_set'.
                                if now_gpu == -1:
                                    if not exist_rec_tmr_results(model, c_num, dataset, setup):    
                                        create_sample_data_mp(dataset=dataset, model=model, client_num=c_num, 
                                                        local_round=l_round, rec_sample=all_set, fed_round=g_round, tmr_flag=1, setup=setup, now_gpu=now_gpu)  
        # all_gpus = 

