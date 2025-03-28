from helper_shap import load_rec_sample_results, handle_sample_time
from helper_alg import *
# where DEF_SHAP, GTB_SHAP, TMC_SHAP, LOO_SHAP : (rec_sample, client_num) --> svacc, svloss, tacc, tloss
# OR_SHAP, TMR_SHAP: (client_num, (dataset, model)) --> svacc, svloss, tacc, tloss
from helper_metric import MAE, MSE

import os
import math
import pickle
TEST_MODE = False
def compute_shapley_test(model, dataset, client_num, algorithm=DEF_SHAP):


    # # Use usr define for test.    
    # rec_sample_res = handle_sample_time()
    # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = DEF_SHAP_P(rec_sample_res, 3)
    # # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = TMC_SHAP(rec_sample_res, 3)
    # # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = GTB_SHAP(rec_sample_res, 3)
    # # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = LOO_SHAP(rec_sample_res, 3)
    # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = CC_SHAP(rec_sample_res, 3)
    # # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = CC_Neyman_SHAP(rec_sample_res, 3)
    # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = DEF_SHAP_P(rec_sample_res, 3)


    # # Load the the acc,loss, time of a subset.
    # # print("rec_sample_res", rec_sample_res)
    # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = DEF_SHAP(rec_sample_res, client_num, output_flag=False)
    # print("ACC_DEF:", SV_approx_acc, "TIME:", SV_time_acc)
    # for k in range(1,client_num+1):
    #     KG_approx_acc, KG_approx_loss, KG_time_acc, KG_time_loss = KG_SHAP(rec_sample_res, client_num, k, output_flag=False)
    #     print("KG[{}] = {}".format(k, KG_approx_acc))
    #     print("TM[{}]={}".format(k, KG_time_acc))
    #     # print("ACC_KG:", KG_approx_acc, "TIME:", KG_time_acc)
    # # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = DEF_SHAP_P(rec_sample_res, client_num) 
    # # TMC_acc, SV_approx_loss, SV_time_acc, SV_time_loss = TMC_SHAP(rec_sample_res, client_num)
    # # # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = GTB_SHAP(rec_sample_res, client_num)
    # # LOO_acc, SV_approx_loss, SV_time_acc, SV_time_loss = LOO_SHAP(rec_sample_res, client_num)
    # # OR_acc, SV_approx_loss, SV_time_acc, SV_time_loss = OR_SHAP(client_num, (dataset, model))
    # # TMR_acc, SV_approx_loss, SV_time_acc, SV_time_loss = TMR_SHAP(client_num, (dataset, model))
    # # # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = GTG_SHAP(client_num, (dataset, model))
    # CC_acc, SV_approx_loss, SV_time_acc, SV_time_loss = CC_SHAP(rec_sample_res, client_num, sample_round=15, output_flag=False)
    # print("CC-SHAP:", CC_acc, SV_time_acc)
    # TMC_acc, SV_approx_loss, SV_time_acc, SV_time_loss = TMC_SHAP(rec_sample_res, client_num,  sample_round=5, ouput_flag=False)
    # print("TMC_acc:", TMC_acc, SV_time_acc)


    # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = CC_Neyman_SHAP(rec_sample_res, client_num)
    # SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss = CC_Berstin_SHAP(rec_sample_res, client_num)
    # print("sv_acc:{} (time={}), sv_loss:{} (time={})".format(SV_approx_acc, SV_time_acc, SV_approx_loss, SV_time_loss))

    # return SV_approx_acc, SV_approx_loss, SV_time_acc, SV_time_loss 
    # print("MSE: DEF{}, ")
    for cnum in [3, 6, 10, 15]:
        rec_sample_res = load_rec_sample_results(model, int(cnum), dataset)    
        acc, loss, time, _ = KG_SHAP(rec_sample_res, int(cnum), int(max(math.sqrt((2)**cnum), 4)))
        print('Srd:', max(math.sqrt(2**cnum), 4), '  #Time:{}'.format(time))
    


def exp_on_real_datasets(train_Setups = ['same'], datasets = ['emnist'], client_nums = [3, 6, 10, 15], used_models = ['cnn_model'], times = 1):
    algs  = dict({"Comb-Shapley":DEF_SHAP, "Perm-Shapley":DEF_SHAP_P, 'Extended-TMC':TMC_SHAP,
                       "OR":OR_SHAP, "$lambda$-MR":TMR_SHAP, "Extended-GTB":GTB_SHAP, "DIG-FL":LOO_SHAP,
                       'CC-Shapley':CC_SHAP, "Light Sampling": KG_SHAP, "GTG-Shapley":GTG_SHAP
    })
    # algs  = dict({"Comb-Shapley":DEF_SHAP, "Perm-Shapley":DEF_SHAP_P, 'Extended-TMC':TMC_SHAP,
    #                    "Extended-GTB":GTB_SHAP, "DIG-FL":LOO_SHAP,
    #                    'CC-Shapley':CC_SHAP, "Light Sampling": KG_SHAP
    # })
    def_shap = ['Comb-Shapley', 'Perm-Shapley', 'DIG-FL']
    samp_shap= ['Extended-TMC', "Extended-GTB", "CC-Shapley"]
    grad_shap= ['GTG-Shapley', 'OR', "$lambda$-MR"]
    ligh_shap= ['Light Sampling']

    algs_res = {}

    for cnum in client_nums:
        for dataset in datasets:
            for set_up in train_Setups:
                for mod in used_models:
                    for t in range(times):
                        exp_path = './expres/'
                        exp_name = mod + '_' + str(cnum) + '_' + dataset + "_" + set_up +'.res'
                        if times > 1:
                            exp_name += "." + str(t)
                        rec_sample_res = load_rec_sample_results(mod, cnum, dataset, set_up)    
                        samp_rds = int(max(math.sqrt(2**cnum), 5))
                        
                        if os.path.exists(exp_path+exp_name):
                            with open(exp_path+exp_name, 'rb') as fin:
                                pre_res = pickle.load(fin)

                        for alg_name, alg_func in algs.items():
                            if alg_name != "Light Sampling":
                                algs_res[alg_name] = pre_res[alg_name]
                                continue
                            if alg_name in def_shap:
                                if alg_name ==  'Perm-Shapley' and cnum > 10:
                                    acc, loss, acc_time, loss_time = [],[],-1,-1
                                    algs_res[alg_name] = (acc, acc_time)
                                else:
                                    acc, loss, acc_time, loss_time = alg_func(rec_sample_res, cnum)
                                    # algs_res[alg_name] = (acc, acc_time)
                                    algs_res[alg_name] = (loss, loss_time)
                            
                            if alg_name in grad_shap:
                                acc, loss, acc_time, loss_time = alg_func(cnum, (dataset, mod, set_up))
                                # algs_res[alg_name] = (acc, acc_time)
                                algs_res[alg_name] = (loss, loss_time)
                            
                            if alg_name in samp_shap:
                                if alg_name == 'Extended-TMC':
                                    acc, loss, acc_time, loss_time = alg_func(rec_sample_res, cnum, samp_rds//cnum)
                                    # algs_res[alg_name] = (acc, acc_time)
                                    algs_res[alg_name] = (loss, loss_time)
                                if alg_name == 'Extended-GTB':
                                    acc, loss, acc_time, loss_time = alg_func(rec_sample_res, cnum, samp_rds)
                                    # algs_res[alg_name] = (acc, acc_time)
                                    algs_res[alg_name] = (loss, loss_time)
                                if alg_name == 'CC-Shapley':
                                    acc, loss, acc_time, loss_time = alg_func(rec_sample_res, cnum, samp_rds // 2)
                                    # algs_res[alg_name] = (acc, acc_time)
                                    algs_res[alg_name] = (loss, loss_time)
                            if alg_name in ligh_shap:
                                acc, loss, acc_time, loss_time = alg_func(rec_sample_res, cnum, samp_rds)
                                # algs_res[alg_name] = (acc, acc_time)
                                algs_res[alg_name] = (loss, loss_time)

                        with open(exp_path+exp_name, 'wb') as fout:
                            pickle.dump(algs_res, fout)
                        # print(exp_path+exp_name,'\n', algs_res)
                    
        # pass


if __name__ == "__main__":
    
    # pass
    if TEST_MODE:
        compute_shapley_test('cnn_model', 'emnist', 5)
        
    else:
        # exp_on_real_datasets(train_Setups = ['same'], datasets = ['emnist'], client_nums = [3, 6, 10, 15], used_models = ['cnn_model'])
        # exp_on_real_datasets(train_Setups = ['same'], datasets = ['emnist'], client_nums = [3, 6, 10, 15], used_models = ['linear_model'])
        

        # exp_on_real_datasets(train_Setups = ['same', 'mixDtr', 'mixSize', 'noiseX', 'noiseY'], datasets = ['mnist'], client_nums = [10], used_models = ['cnn_model'])
        # exp_on_real_datasets(train_Setups = ['same'], datasets = ['mnist'], client_nums = [3], used_models = ['linear_model'], times = 5)
        exp_on_real_datasets(train_Setups = ['same'], datasets = ['chengdu'], client_nums = [10], used_models = ['lstmtraj_model'])

