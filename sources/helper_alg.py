from helper_shap import buildPowerSets, power2number
import scipy.special as scp
import numpy as np
import random
import os
import pickle as pk
import time
import math
import copy


def DEF_SHAP_P(rec_sample, client_num):
    from itertools import permutations  
    def get_permutations(n):
        # return list(permutations(range(n)))
        return permutations(range(n))
    print("Now run the DEF_SHAP_Perm_Version.")
    # all_sets = buildPowerSets([i for i in range(client_num)])
    shapley_value_acc = [0 for _ in range(client_num)]
    shapley_value_loss =[0 for _ in range(client_num)]
    cost_time = 0

    # print("RecSample:", rec_sample)
    all_perm = get_permutations(client_num)
    all_perm_len = math.factorial(client_num)
    # len(all_perm)
    print("Now we compute values under all permutations in Perm-Shapley {}.".format(all_perm_len))
    for perm in all_perm:
        # print("Now Perm is:", perm)
        for ind in range(client_num):
            shapley_value_acc[perm[ind]] = shapley_value_acc[perm[ind]] + (rec_sample[str(power2number(perm[:ind+1]))]['acc'] - rec_sample[str(power2number(perm[:ind]))]['acc'])/all_perm_len
            shapley_value_loss[perm[ind]] = shapley_value_loss[perm[ind]] + (rec_sample[str(power2number(perm[:ind+1]))]['loss'] - rec_sample[str(power2number(perm[:ind]))]['loss'])/(-all_perm_len)
            cost_time = cost_time + rec_sample[str(power2number(perm[:ind+1]))]['time'] + rec_sample[str(power2number(perm[:ind]))]['time']
            # print(cost_time)
            # print("{} --> {}_acc - {}_acc ## now_time={}".format(perm[ind], perm[:ind+1], perm[:ind], cost_time))

    print("DEF_P shapley Value ({}):".format(cost_time), '\n', shapley_value_acc, '\n', shapley_value_loss)        
    
    return shapley_value_acc, shapley_value_loss, cost_time, cost_time

def DEF_SHAP(rec_sample, client_num, output_flag=True):
    if output_flag:
        print("Now run the DEF_SHAP_Weight_Version.")
    all_sets = buildPowerSets([i for i in range(client_num)])
    shapley_value_acc = [0 for _ in range(client_num)]
    shapley_value_loss =[0 for _ in range(client_num)]

    # print("RecSample:", rec_sample)
    for cid in range(client_num):
        sv_acc = 0.0
        sv_loss = 0.0
        for subset in all_sets:
            if not (cid in subset):
                
                delta_acc = rec_sample[str(power2number(subset+[cid]))]['acc'] - rec_sample[str(power2number(subset))]['acc']
                # print("{}_acc - {}_acc = {}, {}, weight_contrib:{}".format(subset+[cid], subset, delta_acc, scp.comb(client_num-1, len(subset)), delta_acc / scp.comb(client_num-1, len(subset))))
                sv_acc = sv_acc + delta_acc / scp.comb(client_num-1, len(subset))

                # print("Compute Loss of subset {}".format(subset))
                delta_loss = -(rec_sample[str(power2number(subset+[cid]))]['loss'] - rec_sample[str(power2number(subset))]['loss'])
                sv_loss = sv_loss + delta_loss / scp.comb(client_num-1, len(subset))             

        shapley_value_acc[cid] = sv_acc / client_num
        shapley_value_loss[cid] = sv_loss / client_num

    cost_time = sum([(rec_sample[str(power2number(subset))]['time']) for subset in all_sets])
    if output_flag:
        print("DEF_W shapley Value ({}):".format(cost_time), '\n', shapley_value_acc, '\n', shapley_value_loss)        
    
    return shapley_value_acc, shapley_value_loss, cost_time, cost_time

def KG_SHAP(rec_sample, client_num, sample_round=100, output_flag=True):

    k, res, rsd = 0, 1, 0
    # sample_round = int(max(math.sqrt(math.sqrt(2**client_num)), 5))
    # use_round = int(max(4, math.sqrt(sample_round)))
    use_round = sample_round
    for i in range(1, client_num):

        if res <= use_round:
            k = i-1
        else:
            rsd = (res+scp.comb(client_num, i))-use_round
            break
        res += scp.comb(client_num, i)
    # k = max()
     
    if output_flag:
        print("Now run the KG_SHAP with k={} in sample_rounds={}, rsd={}.".format(k, use_round, rsd))
    all_sets = buildPowerSets([i for i in range(client_num)])
    shapley_value_acc = [0 for _ in range(client_num)]
    shapley_value_loss =[0 for _ in range(client_num)]    
    cost_time = 0
    # print("RecSample:", rec_sample)
    for cid in range(client_num):
        sv_acc = 0.0
        sv_loss = 0.0
        for subset in all_sets:
            if not (cid in subset) and (len(subset)<k):

                delta_acc = rec_sample[str(power2number(subset+[cid]))]['acc'] - rec_sample[str(power2number(subset))]['acc']
                # print("{}_acc - {}_acc = {}, {}, weight_contrib:{}".format(subset+[cid], subset, delta_acc, scp.comb(client_num-1, len(subset)), delta_acc / scp.comb(client_num-1, len(subset))))
                sv_acc = sv_acc + delta_acc / scp.comb(client_num-1, len(subset))

                # print("Compute Loss of subset {}".format(subset))
                delta_loss = -(rec_sample[str(power2number(subset+[cid]))]['loss'] - rec_sample[str(power2number(subset))]['loss'])
                sv_loss = sv_loss + delta_loss / scp.comb(client_num-1, len(subset))             

        shapley_value_acc[cid] = sv_acc / client_num
        shapley_value_loss[cid] = sv_loss / client_num

    cost_time = sum([(rec_sample[str(power2number(subset))]['time']) for subset in all_sets if (len(subset)<=k)])
    if output_flag:
        print("K-Greedy shapley Value ({}):".format(cost_time), '\n', shapley_value_acc, '\n', shapley_value_loss)        
    return shapley_value_acc, shapley_value_loss, cost_time, cost_time

def IPSS_SHAP(rec_sample, client_num, sample_round, output_flag=True):
    all_sets = buildPowerSets([i for i in range(client_num)])
    sample_sets = []
    for i in range(client_num + 1):
        layer_sets = list(filter(lambda s: len(s) == i, all_sets))
        layer_num = len(layer_sets)
        if sample_round >= layer_num:
            sample_round -= layer_num
            sample_sets += layer_sets
        else:
            sample_sets += random.sample(layer_sets, sample_round)
            break
    sample_indices = [power2number(s) for s in sample_sets]

    shapley_value_acc = [0 for _ in range(client_num)]
    shapley_value_loss = [0 for _ in range(client_num)]

    for cid in range(client_num):
        sv_acc = 0.0
        sv_loss = 0.0
        for subset in sample_sets:
            if not (cid in subset) and (power2number(subset + [cid]) in sample_indices):
                delta_acc = rec_sample[str(power2number(subset+[cid]))]['acc'] - rec_sample[str(power2number(subset))]['acc']
                sv_acc = sv_acc + delta_acc / scp.comb(client_num-1, len(subset))
                delta_loss = -(rec_sample[str(power2number(subset+[cid]))]['loss'] - rec_sample[str(power2number(subset))]['loss'])
                sv_loss = sv_loss + delta_loss / scp.comb(client_num-1, len(subset))             
        shapley_value_acc[cid] = sv_acc / client_num
        shapley_value_loss[cid] = sv_loss / client_num
    
    cost_time = sum([(rec_sample[str(power2number(subset))]['time']) for subset in sample_sets])
    if output_flag:
        print("IPSS shapley Value ({}):".format(cost_time), '\n', shapley_value_acc, '\n', shapley_value_loss)        
    return shapley_value_acc, shapley_value_loss, cost_time, cost_time

def TEST_SHAP(rec_sample, client_num, sample_round, output_flag=True):
    all_sets = buildPowerSets([i for i in range(client_num)])
    sample_sets = []
    # weight = [1 / scp.comb(client_num, k) / (k + 1) for k in range(client_num + 1)]
    # idx = np.argsort(weight)[::-1]
    idx = [0, 1, 10, 9, 2, 3, 8, 7, 4, 5, 6]
    for i in idx:
        layer_sets = list(filter(lambda s: len(s) == i, all_sets))
        layer_num = len(layer_sets)
        if sample_round >= layer_num:
            sample_round -= layer_num
            sample_sets += layer_sets
        else:
            sample_sets += random.sample(layer_sets, sample_round)
            break
    sample_indices = [power2number(s) for s in sample_sets]

    shapley_value_acc = [0 for _ in range(client_num)]
    # shapley_value_loss = [0 for _ in range(client_num)]

    for cid in range(client_num):
        sv_acc = 0.0
        sv_loss = 0.0
        for subset in sample_sets:
            if (cid not in subset) and (power2number(subset + [cid]) in sample_indices):
                delta_acc = rec_sample[str(power2number(subset+[cid]))]['acc'] - rec_sample[str(power2number(subset))]['acc']
                sv_acc = sv_acc + delta_acc / scp.comb(client_num-1, len(subset))
                # delta_loss = -(rec_sample[str(power2number(subset+[cid]))]['loss'] - rec_sample[str(power2number(subset))]['loss'])
                # sv_loss = sv_loss + delta_loss / scp.comb(client_num-1, len(subset))             
        shapley_value_acc[cid] = sv_acc / client_num
        # shapley_value_loss[cid] = sv_loss / client_num
    
    cost_time = sum([(rec_sample[str(power2number(subset))]['time']) for subset in sample_sets])
    # if output_flag:
    #     print("IPSS shapley Value ({}):".format(cost_time), '\n', shapley_value_acc, '\n', shapley_value_loss)        
    return shapley_value_acc, 0, cost_time, cost_time

def K2G_SHAP(rec_sample, client_num, sample_round=100, output_flag=True):
    k = 2
    all_sets = buildPowerSets([i for i in range(client_num)])
    shapley_value_acc = [0 for _ in range(client_num)]
    shapley_value_loss =[0 for _ in range(client_num)]    
    cost_time = 0
    for cid in range(client_num):
        sv_acc = 0.0
        sv_loss = 0.0
        for subset in all_sets:
            if not (cid in subset) and (len(subset)<k):
                delta_acc = rec_sample[str(power2number(subset+[cid]))]['acc'] - rec_sample[str(power2number(subset))]['acc']
                sv_acc = sv_acc + delta_acc / scp.comb(client_num-1, len(subset))
                delta_loss = -(rec_sample[str(power2number(subset+[cid]))]['loss'] - rec_sample[str(power2number(subset))]['loss'])
                sv_loss = sv_loss + delta_loss / scp.comb(client_num-1, len(subset))             
        shapley_value_acc[cid] = sv_acc / client_num
        shapley_value_loss[cid] = sv_loss / client_num
    cost_time = sum([(rec_sample[str(power2number(subset))]['time']) for subset in all_sets if (len(subset)<=k)])
    if output_flag:
        print("2-Greedy shapley Value ({}):".format(cost_time), '\n', shapley_value_acc, '\n', shapley_value_loss)        
    return shapley_value_acc, shapley_value_loss, cost_time, cost_time

def TMC_SHAP(rec_sample, client_num, sample_round=30, ouput_flag=True):
    def comp(rec_sample, client_num, utils='acc', sample_round=100):
        # all_sets = buildPowerSets(client_num)
        shapley_value_utility = [0 for i in range(client_num)]
        shapley_value_loss =[0 for i in range(client_num)]

        now_perm = [i for i in range(client_num)]
        cost_time_utility = rec_sample[str(power2number(now_perm))]['time']
        # cost_time_loss= rec_sample[str(power2number(now_perm))]['time']

        # hyper_parameters in trunced_monte_carlo_method.
        convergen_rate = 0.001
        perm_delta = 0.0001 
        total_round = 0
        convergen_flag = False
        allset_acc = rec_sample[str(power2number(now_perm))][utils]
        temp_utility = [0 for _ in range(client_num+1)]

        # adopt acc as the utility function 
        while (not convergen_flag) and  (total_round < sample_round):
            # if total_round% 10==0: print("R{}".format(total_round))
            np.random.shuffle(now_perm)
            total_round = total_round + 1
            # print('Round#{}#: NowPerm is {}'.format(total_round, now_perm))

            last_sv_acc = copy.deepcopy(shapley_value_utility) 
            temp_utility[0] = rec_sample[str(power2number([]))][utils]
            for j in range(0, client_num):
                sub_perm = now_perm[:j+1]
                # print("   subperm is {}".format(sub_perm), 'perm_delat is {}'.format(perm_delta))
                if np.abs(allset_acc- temp_utility[j]) < perm_delta:
                    temp_utility[j+1] = temp_utility[j]
                else:
                    sub_perm_index = str(power2number(sub_perm))
                    temp_utility[j+1] = rec_sample[sub_perm_index][utils]
                    cost_time_utility += rec_sample[sub_perm_index]['time']

                # print("\tSubPerm is {}, Update SubPerm[j]={}".format(sub_perm, sub_perm[j]), )
                shapley_value_utility[sub_perm[j]] = (total_round-1)/total_round * shapley_value_utility[sub_perm[j]] + (1/total_round)*(temp_utility[j+1]-temp_utility[j])
            now_coverage = [abs(shapley_value_utility[_] - last_sv_acc[_]) for _ in range(len(last_sv_acc))]
            if sum(now_coverage) < convergen_rate or total_round > sample_round:
                # print("({}/{})".format(total_round, sample_round))
                convergen_flag = True

        if utils == 'loss':
            shapley_value_utility = [-1*sv for sv in shapley_value_utility]
        return shapley_value_utility, cost_time_utility     
    
    # print("Now run the TMC_SHAP. with output={}".format(ouput_flag))

    shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc', min(2**client_num, sample_round))
    shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss', min(2**client_num, sample_round))

    if ouput_flag:
        print("TMC shapley Value ({} {}):".format(cost_time_acc, cost_time_loss), '\n', shapley_value_acc, '\n', shapley_value_loss)        

    return shapley_value_acc, shapley_value_loss, cost_time_acc, cost_time_loss     

def GTB_SHAP(rec_sample, client_num, sample_round=100, ouput_flag=True):
    import cvxpy as cp
    def isNotConverge(last_u, u, client_num, ROUND_CONVERGE=sample_round, CONVERGE_CRITERIA=0.1):
        if len(last_u) <= ROUND_CONVERGE:
            return True
        for i in range(0, ROUND_CONVERGE):
            ele = last_u[i]
            delta = np.sum(np.abs(u - ele), axis=(0, 1)) / client_num
            if delta > CONVERGE_CRITERIA:
                return True
        return False
    
    def solveF(U, u_all):
        def solve_feasibility(U, u_all, exps=1):
            n = U.shape[0]
            x = cp.Variable(n)
            constraints = []
            # print("n is {}, x is {}".format(n, x))

            # print("U in solve_F:", U)
            for i in range(n):
                constraints.append(x[i]>=0)
                for j in range(n):
                    constraints.append(cp.abs(x[i] - x[j] - U[i][j])<=exps)
            constraints.append(cp.sum(x) == u_all)

            # print("Constraints is {}".format(constraints))
            objective = cp.Minimize(0)
            prob = cp.Problem(objective, constraints)
            prob.solve()

            # print(prob.status)
            if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
                return x.value
            else:
                return []

        # relax the constrain utill we get fesible results.
        exps = 1/(2*math.sqrt(U.shape[0]))
        res = []
        while (len(res) == 0) and (exps< 10):
            res = solve_feasibility(U, u_all, exps) 
            exps = exps*1.1
  
        if len(res) == 0: return [0 for _ in range(U.shape[0])]        
        return res
    
    # now_perm = [i for i in range(client_num)]
    def comp(rec_sample, client_num, utils, sample_round=100):
        time_cost = 0
        Z = 2*sum([1/k + 1/(client_num-k) for k in range(1, client_num-1)])
        q_values = [1/Z*(1/k+1/(client_num-k)) for k in range(1, client_num)]
        k_values = np.arange(1, client_num)
        U = np.zeros([client_num, client_num], dtype=np.float32)
        U_t = [0 for _ in range(sample_round)] 

        total_round = 0
        beta = [[0 for _ in range(client_num)] for __ in range(sample_round+1)]
        

        while total_round < sample_round:
            round_k =  np.random.choice(k_values, size=100, p=q_values/np.sum(q_values))[0]
            k_t = 2
            random_subset = []
            for j in range(k_t):   
                new_subset = random.sample(range(0, client_num), round_k)
                for cid in random_subset:
                    beta[total_round][cid] = 1
                random_subset = list(set(random_subset+new_subset))

            random_subset_index = str(power2number(random_subset))
            if utils == 'loss':
                U_t[total_round] = -rec_sample[random_subset_index][utils]
            else:
                U_t[total_round] = rec_sample[random_subset_index][utils]            
            time_cost += rec_sample[random_subset_index]['time']
            total_round += 1

        # print("U_t is {}, Z is {}, time_cost is {}".format(U_t, Z, time_cost))
        for i in range(0, client_num):
            for j in range(i, client_num):
                    U[i][j] = Z/sample_round*sum([U_t[t]*(beta[t][i]-beta[t][j]) for t in range(total_round)])

        allset_index = str(power2number([i for i in range(0, client_num)]))
        U_D = rec_sample[allset_index][utils]
        time_cost += rec_sample[allset_index]['time']  
        shapley_value = list(solveF(U, U_D))


        return shapley_value, time_cost
    
    # shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc', min(sample_round, 2**client_num))
    # shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss', min(sample_round, 2**client_num))
    
    shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc', min(sample_round, 2**client_num))
    shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss', min(sample_round, 2**client_num))
    if ouput_flag:
        print("GTB shapley Value ({}):".format(cost_time_acc), '\n', shapley_value_acc, '\n', shapley_value_loss)        

    return shapley_value_acc, shapley_value_loss, cost_time_acc, cost_time_loss 

def LOO_SHAP(rec_sample, client_num):

    def comp(rec_sample, client_num, utils):

        allset = [i for i in range(client_num)]
        # time_cost = 0
        allset_index  = str(power2number(allset))
        allset_utility = rec_sample[allset_index][utils]
        time_cost = rec_sample[allset_index]['time']

        loo_value = [0 for i in range(client_num)]
        for cid in range(client_num):
            subset = list(set(allset)-set([cid]))
            subset_index = str(power2number(subset))
            loo_value[cid] = allset_utility - rec_sample[subset_index][utils]
            time_cost += rec_sample[subset_index]['time']
        if utils == 'loss':
            loo_value = [-sv for sv in loo_value]
        return loo_value, time_cost

    shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc')
    shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss')
    print("LOO shapley Value ({}):".format(cost_time_acc), '\n', shapley_value_acc, '\n', shapley_value_loss)        

    return shapley_value_acc, shapley_value_loss, cost_time_acc, cost_time_loss     

def OR_SHAP(client_num, file_para):
    from fed_models import FED_MODEL_DICT
    from fed_models import FED_SHAPE_DICT
    import re

    def get_file_list(client_num, file_para):
        # import re
        # print("#In get_file_list#: with file_para={}".format(file_para))

        file_list = os.listdir('./rec_fed_grad/')
        # print("#In get_file_list#--> file_list is {}".format(file_list))

        dataset, model, train_setup = file_para
        grad_path = list(filter(lambda x: x.startswith(model) and x.endswith(dataset+'_'+train_setup) and str(client_num) in x, file_list))[0]
        # print("#In get_file_list#--> grad_path is {}".format(grad_path))
        
        grad_file_list = os.listdir('./rec_fed_grad/'+grad_path)
        # print("#In get_file_list#--> grad_file_list is {}".format(grad_file_list))

        # grad_path = 'linear_model_10_emnist', so './rec_fed_grad/'+grad_path+'/' --> './rec_fed_grad/linear_model_10_emnist/'
        fed_round_cand = list(filter(lambda x: '_' in x, grad_file_list))
        # print("#In get_file_list#--> fed_round_cand is {}".format(fed_round_cand))

        round_numbers = [re.findall(r'\d+?\d*', x) for x in fed_round_cand]
        fed_round = max([int(x[1]) for x in round_numbers])
        # print("#In get_file_list#--> fed_round is {}".format(fed_round))
        
        # print("#In get_file_list#-->{}, {}".format(fed_round, './rec_fed_grad/'+grad_path+'/'))
        return fed_round, './rec_fed_grad/'+grad_path+"/"

    def load_grad(file_path, cid, round, subset):
        if not (cid in subset): return [] 
        file_name = file_path+str(cid)+'_'+str(round)+'.grad_rec'
        with open(file_name, "rb") as fin:
            grad = pk.load(fin)
        return grad
        # return pk.load(file_name, 'rb')


    # def get_or_name_rec(dir, model_name, client_num, datasets_name):
    #     return dir + 'or_' + model_name + '_' + str(client_num) + '_' + datasets_name + '.rec'
    
    # load_the_approx_one_round_results of subsets.
    def load_or_rec_sample_results(model_name, client_num, datasets_name, setup_name):
        # file_name_rec_time = get_or_name_rec("./rec_fed_sample_time/", args_model, args_client_num, args_dataset)
        file_name_rec_time = "./rec_fed_sample_time/or_" + model_name + '_' + str(client_num) + '_' + datasets_name + '_' + setup_name + '.rec'
        with open(file_name_rec_time, "rb") as fin:
            now_rec = pk.load(fin)
        return now_rec

    def rec_sample_or_time(args, accuarcy, loss, time_slot):
        
        client_num, model_name, datasets_name, setup_name, subset = args 

        # check the dir path. 
        if not os.path.exists("./rec_fed_sample_time/"): os.mkdir("./rec_fed_sample_time/")

        or_name_rec = "./rec_fed_sample_time/or_" + model_name + '_' + str(client_num) + '_' + datasets_name +'_' + setup_name + '.rec'
            
        # check the file path.
        if not os.path.exists(or_name_rec):
            # print("now we create the " + or_name_rec)
            with open(or_name_rec, 'wb') as fout: 
                blank_dict = {}
                pk.dump(blank_dict, fout)

        # load the or_rec_grad_file.
        with open(or_name_rec, "rb") as fin:
            now_rec = pk.load(fin)

        # save the or_rec_grad_file.
        now_rec[str(power2number(subset))] = {'loss':loss, 'acc': accuarcy, 'time': time_slot}
        with open(or_name_rec, 'wb') as fout:
            pk.dump(now_rec, fout)


    def approx_weights(subset, client_num, file_para, global_model_weights):
        

        if len(subset) == 0:
            return global_model_weights
        
        # dataset, model = file_para
        # print("#In approx_weights# with subset={}, n={}, file_para={}".format(subset, client_num, file_para))

        fed_round, file_grad_path = get_file_list(client_num, file_para)
        # print("#In approx_weights#  fed_round={}, file_grad_path={}".format(fed_round, file_grad_path))

        subset_datasize = np.array([0 for _ in range(client_num)])
        for cid in subset:
            with open(file_grad_path + str(cid)+'.datasize', 'rb') as fin:
                # print("#In approx_weights# open {}".format(file_grad_path + str(cid)+'.datasize'))
                subset_datasize[cid] = pk.load(fin)
        # print("#In approx_weights# We have loaded the datasize.")

        all_datasize = np.sum(subset_datasize)
        grad_alpha = (subset_datasize/all_datasize)
        # print("#In approx_weights# all_datasize is={}, grad_alpha={}.".format(all_datasize, grad_alpha))
       

        # get the Grad[round][cid]
        subset_rec_grad = [[] for _ in range(fed_round)]
        for round in range(fed_round):
            subset_rec_grad[round] = [load_grad(file_grad_path, cid, round, subset) for cid in range(client_num)]

        # re-construct the grad  
        # temp = [np.zeros(gmw_layer.shape) for gmw_layer in global_model_weights]
        for round in range(fed_round):
            for layers in range(len(global_model_weights)):
                for cid in subset:
                    # exec aggregation on each layer
                    global_model_weights[layers] += (grad_alpha[cid]*subset_rec_grad[round][cid][layers])

        # print("#In approx_weights# Re-construct Grad.")

        return global_model_weights

    def create_one_round(client_num, file_para):
        dataset, model, setup_name = file_para
        # or_name_rec = get_or_name_rec("./rec_fed_sample_time/", model, client_num, dataset)
        or_name_rec = "./rec_fed_sample_time/or_" + model + '_' + str(client_num) + '_' + dataset + '_' + setup_name + '.rec'
        # print("#create_one_round# --> or_name_rec is ", or_name_rec)

        if os.path.exists(or_name_rec):
            now_rec = load_or_rec_sample_results(model, client_num, dataset, setup_name)
            # print("------now_rec_of_or_rec-----")
            # print(now_rec)
            return now_rec
        # else:
        #     print("Error! You do not have the client!")
        #     return {}

        # load the test data
        # print("Load the test data for OR_SHAP approximation")
        data_path = './datasets/'+str(dataset)+'/client_'+str(client_num)+"_"+ setup_name +"/"
        # print("Data Path is", data_path)
        testX = pk.load(open(data_path+'testX.pk', 'rb'))
        testY = pk.load(open(data_path+'testY.pk', 'rb'))

        # compute the approx utils
        all_sets = buildPowerSets([i for i in range(client_num)])
        data_shape = FED_SHAPE_DICT[dataset]
        global_model = FED_MODEL_DICT[model](data_shape[0], data_shape[1])
        global_weights = global_model.model_get_weights()
        
        # compute the approx acc and time cost
        # subset_acc_dict, subset_loss_dict, time_cost = {}, {}, 0
        # print("All set is ", all_sets)
        for subset in all_sets:
            subset_global_weights = copy.deepcopy(global_weights)
            # print("Now we approx {} in {}".format(subset, [i for i in range(client_num)]))
            # compute the approx model
            begin_time = time.perf_counter()
            subset_global_weights = approx_weights(subset, client_num, file_para, subset_global_weights)
            
            # compute the approx accuaracy and loss
            global_model.model_load_weights(subset_global_weights)
            if dataset == "chengdu":
                subset_loss, subset_acc = global_model.model_get_eval(testX, testY), 0
            else:
                subset_loss, subset_acc = global_model.model_get_eval(testX, testY)
            print("#subset={}#  acc={}, loss={}".format(subset, subset_acc, subset_loss))
            # print('_______', stop)
            end_time = time.perf_counter()
            rec_sample_or_time((client_num, model, dataset, setup_name, subset), subset_acc, subset_loss, end_time-begin_time)
        
        now_rec = load_or_rec_sample_results(model, client_num, dataset, setup_name)
        # print("Now_Rec is:")
        return now_rec

    # print("Now We Start Compute the OR_SHAP.")
    now_rec = create_one_round(client_num, file_para)
    shapley_value_acc, shapley_value_loss, time_cost, time_cost = DEF_SHAP(now_rec, client_num, False)
    print("One-Round(OR) shapley Value ({}):".format(time_cost), '\n', shapley_value_acc, '\n', shapley_value_loss)        

    # shapley_value_acc, shapley_value_loss, time_cost = comp(client_num, file_para)
    return shapley_value_acc, shapley_value_loss, time_cost, time_cost

def TMR_SHAP(client_num, file_para):

    # parameters 
    TMR_LAMBDA = 0.98

    # def comp(client_num, file_para):
    dataset, model, setup_name = file_para
    file_name_tmr = "./rec_fed_tmr_res/" + 'tmr_' + model + '_' + str(client_num) + '_' + dataset + '_' + setup_name + '.rec'
    # print("#In TMR_SHAP#", file_name_tmr)
    
    with open(file_name_tmr, "rb") as fin:
        tmr_rec = pk.load(fin)     

    shapley_value_acc = [0 for _ in range(client_num)]
    shapley_value_loss =[0 for _ in range(client_num)]
    time_cost = 0
    for round in range(len(tmr_rec)):
        rsv_acc, rsv_loss, rsv_time, _ = DEF_SHAP(tmr_rec[round], client_num, False)
        for cid in range(client_num): 
            shapley_value_acc[cid] += (TMR_LAMBDA**round)*rsv_acc[cid]
        for cid in range(client_num):
            shapley_value_loss[cid]+= (TMR_LAMBDA**round)*rsv_loss[cid]
        time_cost += rsv_time
    
    print("Multi-Round(MR) shapley Value ({}):".format(time_cost), '\n', shapley_value_acc, '\n', shapley_value_loss)        
    return shapley_value_acc, shapley_value_loss, time_cost, time_cost

def GTG_SHAP(client_num, file_para):


    dataset, model, setup_name = file_para
    file_name_tmr = "./rec_fed_tmr_res/" + 'tmr_' + model + '_' + str(client_num) + '_' + dataset + '_' + setup_name + '.rec'
    
    with open(file_name_tmr, "rb") as fin:
        tmr_rec = pk.load(fin)     

    shapley_value_acc = [0 for i in range(client_num)]
    shapley_value_loss =[0 for i in range(client_num)]
    cost_time_loss = 0
    cost_time_acc = 0
    for round in range(len(tmr_rec)):
        # rsv_acc, rsv_loss, rsv_time, _ = DEF_SHAP(tmr_rec[round], client_num)
        rsv_acc, rsv_loss, rsv_time_acc, rsc_time_loss = TMC_SHAP(tmr_rec[round], client_num, int(max(math.sqrt(2**client_num), 5)), False)
        for cid in range(client_num): 
            shapley_value_acc[cid] += rsv_acc[cid]
        for cid in range(client_num):
            shapley_value_loss[cid]+= rsv_loss[cid]
        cost_time_acc += rsv_time_acc
        cost_time_loss += rsc_time_loss
    print("GTG shapley Value ({} {}):".format(cost_time_acc, cost_time_loss), '\n', shapley_value_acc, '\n', shapley_value_loss)        
    return shapley_value_acc, shapley_value_loss, cost_time_acc, cost_time_loss

def CC_SHAP(rec_sample, client_num, sample_round=150, output_flag=True):
    # compute the shapley value by complementary contributions
    def comp(rec_sample, client_num, utils='acc', sample_round=100):
        shapley_value = [0 for i in range(client_num)]
        time_cost = 0
        now_perm = [i for i in range(client_num)]
        
        temp_rsv = np.zeros((client_num, client_num + 1))
        temp_cnt = np.zeros((client_num, client_num + 1))
        for round in range(sample_round):
            random.shuffle(now_perm)
            num = random.randint(1, client_num)
            # choices = list(range(1, client_num + 1))
            # num = random.choices(choices, weights=[math.comb(client_num, i) for i in choices], k=1)[0]
            l_subset = [now_perm[i] for i in range(num)]
            r_subset = [now_perm[i] for i in range(num, client_num)]

            l_perm_index = str(power2number(l_subset))
            r_perm_index = str(power2number(r_subset))
            time_cost = time_cost + rec_sample[l_perm_index]['time'] + rec_sample[r_perm_index]['time']

            delta_utility = rec_sample[l_perm_index][utils] - rec_sample[r_perm_index][utils]

            for cid in l_subset:
                temp_rsv[cid][num] += delta_utility
                temp_cnt[cid][num] += 1
            
            for cid in r_subset:
                temp_rsv[cid][client_num-num] -= delta_utility
                temp_cnt[cid][client_num-num] += 1
        
        usr_div = lambda x, y: 0 if abs(y) < 1e-8 else x / y
        for cid in range(client_num):
            shapley_value[cid] = 1 / client_num * sum(map(usr_div, temp_rsv[cid], temp_cnt[cid]))
        
        if utils == 'loss':
            shapley_value = [-sv for sv in shapley_value]
        return shapley_value, time_cost

    shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc', min(sample_round, 2**client_num))
    shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss', min(sample_round, 2**client_num))
    if output_flag:
        print("CC shapley Value ({}):".format(cost_time_acc), '\n', shapley_value_acc, '\n', shapley_value_loss)        

    return shapley_value_acc, shapley_value_loss, cost_time_acc, cost_time_loss 
    # pass

def CC_Weight_SHAP(rec_sample, client_num, sample_round=150, output_flag=True):
    # compute the shapley value by complementary contributions
    def comp(rec_sample, client_num, utils='acc', sample_round=100):
        shapley_value = [0 for i in range(client_num)]
        time_cost = 0
        now_perm = [i for i in range(client_num)]
        
        temp_rsv = np.zeros((client_num, client_num + 1))
        temp_cnt = np.zeros((client_num, client_num + 1))
        for round in range(sample_round):
            random.shuffle(now_perm)
            choices = list(range(1, client_num + 1))
            num = random.choices(choices, weights=[math.sqrt(math.sqrt(math.comb(client_num, i))) for i in choices], k=1)[0]
            l_subset = [now_perm[i] for i in range(num)]
            r_subset = [now_perm[i] for i in range(num, client_num)]

            l_perm_index = str(power2number(l_subset))
            r_perm_index = str(power2number(r_subset))
            time_cost = time_cost + rec_sample[l_perm_index]['time'] + rec_sample[r_perm_index]['time']

            delta_utility = rec_sample[l_perm_index][utils] - rec_sample[r_perm_index][utils]

            for cid in l_subset:
                temp_rsv[cid][num] += delta_utility
                temp_cnt[cid][num] += 1
            
            for cid in r_subset:
                temp_rsv[cid][client_num-num] -= delta_utility
                temp_cnt[cid][client_num-num] += 1
        
        usr_div = lambda x, y: 0 if abs(y) < 1e-8 else x / y
        for cid in range(client_num):
            shapley_value[cid] = 1 / client_num * sum(map(usr_div, temp_rsv[cid], temp_cnt[cid]))
        
        if utils == 'loss':
            shapley_value = [-sv for sv in shapley_value]
        return shapley_value, time_cost


    sample_round //= 2
    shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc', sample_round)
    # shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss', min(sample_round, 2**client_num))
    # if output_flag:
    #     print("CC shapley Value ({}):".format(cost_time_acc), '\n', shapley_value_acc, '\n', shapley_value_loss)        

    return shapley_value_acc, 0, cost_time_acc, 0

def CC_Neyman_SHAP(rec_sample, client_num, sample_round=100):
    def comp(rec_sample, client_num, utils='acc', sample_round=20):
        def usr_div(x,y):
            if abs(y)<1e-8: return 0
            return x/y
        def sample_weight(sigma, n, j):
            if j < int(client_num/2): return 0
            up_val = math.sqrt(sum([sigma[i][j]/(j+1)+sigma[i][n-j]/(n-j+1) for i in range(n)]))
            do_val = sum([math.sqrt(sum([sigma[i][k]/k+ sigma[i][n-k]/(n-k+1) for i in range(n)])) for k in range(int(n/2),n)]) 
            return up_val/do_val

        shapley_value = [0 for _ in range(client_num)]
        time_cost, c, m_init, m_all = 0, -1, max(2, int(sample_round/(client_num**2))), sample_round
        now_perm = [_ for _ in range(client_num)]
    
        temp_rsv = np.zeros((client_num, client_num))
        temp_cnt = np.zeros((client_num, client_num))
        sigma_set = [[[] for _ in range(client_num)] for _ in range(client_num)]
        round = 0
        
        while (abs(c-sum(temp_cnt[0][1:])) > 1e-6):
            round+= 1
            c = sum(temp_cnt[0][1:])
            for i in range(0, client_num):
                for j in range(1, client_num):
                    if temp_cnt[i][j] >= m_init: break

                    np.random.shuffle(now_perm)
                    pos = now_perm.index(i)
                    new_perm = now_perm[:pos]+now_perm[pos+1:]
                    np.random.shuffle(new_perm)
                    now_perm = [i] + new_perm

                    l_subset = [now_perm[_] for _ in range(j)]
                    r_subset =[now_perm[_] for _ in range(j, client_num)] 
                    l_perm_index = str(power2number(l_subset))
                    r_perm_index = str(power2number(r_subset))
                    delta_utility = rec_sample[l_perm_index][utils] - rec_sample[r_perm_index][utils] 

                    for cid in range(0, j):
                        temp_rsv[now_perm[cid]][j] += delta_utility
                        temp_cnt[now_perm[cid]][j] += 1
                        sigma_set[now_perm[cid]][j].append(delta_utility)
                    
                    for cid in range(j, client_num):
                        temp_rsv[now_perm[cid]][client_num-j-1] -= delta_utility
                        temp_cnt[now_perm[cid]][client_num-j-1] += 1
                        sigma_set[now_perm[cid]][client_num-j-1].append(delta_utility)
                    
                    time_cost = time_cost + rec_sample[l_perm_index]['time'] + rec_sample[r_perm_index]['time']

        for cid in range(client_num):
            shapley_value[cid] = sum([usr_div(temp_rsv[cid][_],temp_cnt[cid][_]) for _ in range(client_num)]) 

        # print("We show sigma_set.")
        for i in range(len(sigma_set)):
            for j in range(len(sigma_set[0])):
                if len(sigma_set[i][j])==0:
                    sigma_set[i][j] = [0]

        sigma = [[np.var(sigma_set[i][j]) for j in range(0, client_num)] for i in range(client_num)]
        m_first = sum(temp_cnt[0])
        m_list = [sample_weight(sigma, client_num, j) for j in range(0, client_num)]
        m_list = [int(x*(m_all-m_first)) for x in m_list]

        # print('Strat Neyman Sampling:-----------------------------.')
        for j in range(int(client_num/2), client_num-1):
            for k in range(m_list[j]):
                np.random.shuffle(now_perm) 
                l_subset = now_perm[:j]
                r_subset = now_perm[j:]
                l_perm_index = str(power2number(l_subset))
                r_perm_index = str(power2number(r_subset))
                delta_utility = rec_sample[l_perm_index][utils] - rec_sample[r_perm_index][utils]                     
                for cid in l_subset:
                    temp_rsv[cid][j] += delta_utility
                    temp_cnt[cid][j] += 1
                    sigma_set[cid][j].append(delta_utility)
                for cid in r_subset:
                    temp_rsv[cid][client_num-j] -= delta_utility
                    temp_cnt[cid][client_num-j] += 1
                    sigma_set[cid][client_num-j].append(delta_utility)
                
                time_cost = time_cost + rec_sample[l_perm_index]['time'] + rec_sample[r_perm_index]['time']        
        for cid in range(client_num):
            shapley_value[cid] = sum([usr_div(temp_rsv[cid][_],temp_cnt[cid][_]) for _ in range(client_num)])
        if utils == 'loss':
            shapley_value = [-sv for sv in shapley_value]
        return shapley_value, time_cost
    

    shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc', sample_round)
    shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss', sample_round)
    print("CC_Neyman shapley Value ({}):".format(cost_time_acc), '\n', shapley_value_acc, '\n', shapley_value_loss)        

    return shapley_value_acc, shapley_value_loss, cost_time_acc, cost_time_loss 

def CC_Berstin_SHAP(rec_sample, client_num, sample_round=100):
    def comp(rec_sample, client_num, utils='acc', sample_round=20):
        def usr_div(x,y):
            if abs(y)<1e-8: return 0
            return x/y
        def get_epsilon(sigma, n,sigma_set):
            epsilon = [[usr_div(np.sqrt(2*sigma[i][j]*5), len(sigma_set[i][j]))+ usr_div(3*10, len(sigma_set[i][j])) for j in range(n)] for i in range(n)]
            return epsilon
    
        def sample_select(n, epsilon, utils):
            delta = [[] for i in range(n)]
            # delta = [[epsilon[i][j]-epsilon[i][n-j] for j in range(1, math.floor(n/2)+1)] for i in range(n)]  # [1,2]
            for i in range(n):
                delta[i]= ([epsilon[i][j]-epsilon[i][n-j] for j in range(1, math.floor(n/2)+1)])

            l_subset_max, r_subset_max = [],[]
            eps_max = sum([epsilon[i][n-1] for i in range(n)])
            for k in range(math.ceil(n/2), n-1):
                l_subset = []
                u = max([delta[i][n-(k+1)] for i in range(n)]) 
            
                for i in range(n):
                    if delta[i][n-(k+1)] > u:
                        l_subset.append(i)
                i = 1
                while len(l_subset)<(n-(k+1)) and i<n:
                    if abs(delta[i][n-(k+1)]-u)<1e-8:
                        l_subset.append(i)
                    i+=1
                # epsilon_new = copy.deepcopy(epsilon)
                eps_lr = 0
                for cid in l_subset:
                    eps_lr += epsilon[cid][n-(k+1)]
                r_subset = list(set([_ for _ in range(client_num)])-set(l_subset))
                for cid in r_subset:
                    eps_lr += epsilon[cid][k]
                
                if eps_lr>eps_max:
                    eps_max = eps_lr
                    l_subset_max = l_subset
                    r_subset_max = r_subset

            l_perm_index = str(power2number(l_subset_max))
            r_perm_index = str(power2number(r_subset_max))
            delta_utility = rec_sample[l_perm_index][utils] - rec_sample[r_perm_index][utils]
            sample_time_cost = rec_sample[l_perm_index]['time'] + rec_sample[r_perm_index]['time']     
            
            return   delta_utility, l_subset_max, r_subset_max, sample_time_cost

        # print('Start CCB Sampling:----------------A001-------------.')
        shapley_value = [0 for i in range(client_num)]
        time_cost, c, m_init, m_all = 0, -1, max(2, int(sample_round/(client_num**2))), sample_round
        now_perm = [i for i in range(client_num)]
        temp_rsv = np.zeros((client_num, client_num))
        temp_cnt = np.zeros((client_num, client_num))
        sigma_set = [[[] for _ in range(client_num+1)] for _ in range(client_num)]
        

        while abs(c-sum(temp_cnt[0][1:])) > 1e-6:
            c = sum(temp_cnt[0][1:])
            for i in range(client_num):
                for j in range(1,client_num-1):
                    if temp_cnt[i][j] >= 2: break
                    # now_perm = [i for i in range(client_num)]
                    np.random.shuffle(now_perm) 
                    l_subset = now_perm[:j]
                    r_subset = now_perm[j:]
                    l_perm_index = str(power2number(l_subset))
                    r_perm_index = str(power2number(r_subset))
                    delta_utility = rec_sample[l_perm_index][utils] - rec_sample[r_perm_index][utils]
                    
                    for cid in l_subset:
                        temp_rsv[cid][j] += delta_utility
                        temp_cnt[cid][j] += 1
                        sigma_set[cid][j].append(delta_utility)
                    
                    for cid in r_subset:
                        temp_rsv[cid][client_num-j] -= delta_utility
                        temp_cnt[cid][client_num-j] += 1
                        sigma_set[cid][client_num-j].append(delta_utility)
                    
                    time_cost = time_cost + rec_sample[l_perm_index]['time'] + rec_sample[r_perm_index]['time']


        # compute the sigma_ij 
        sigma_set = [[s if len(s)>0 else [0] for s in sigma_set[_]] for _ in range(len(sigma_set))]
        sigma = [[np.var(sigma_set[i][j]) for j in range(client_num)] for i in range(client_num)]
        epsilon = get_epsilon(sigma, client_num, sigma_set)
                
        for k in range(int(m_all-sum(temp_cnt[0]))):
            delta_utility, l_subset, r_subset, sample_time = sample_select(client_num, epsilon, utils)
            for cid in l_subset:
                # if not (client_num-len(l_subset) in range(0,client_num)): break
                temp_rsv[cid][len(l_subset)] += delta_utility
                temp_cnt[cid][len(l_subset)] += 1
                sigma_set[cid][j].append(delta_utility)    
            for cid in r_subset:
                if not (client_num-len(l_subset) in range(0,client_num)): break
                temp_rsv[cid][client_num-len(l_subset)] -= delta_utility
                temp_cnt[cid][client_num-len(l_subset)] += 1
                sigma_set[cid][client_num-j].append(delta_utility)
            
            time_cost += sample_time        
        for cid in range(client_num):
            shapley_value[cid] = sum([usr_div(temp_rsv[cid][_],temp_cnt[cid][_]) for _ in range(client_num)])
        if utils == 'loss':
            shapley_value = [-sv for sv in shapley_value]
        return shapley_value, time_cost
    
    shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc', sample_round)
    shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss', sample_round)
    print("CC_Berstin_SHAP shapley Value ({}):".format(cost_time_acc), '\n', shapley_value_acc, '\n', shapley_value_loss)        

    return shapley_value_acc, shapley_value_loss, cost_time_acc, cost_time_loss 

def SS_SHAP(rec_sample, client_num, sample_round, output_flag, get_pair):
    all_sets = buildPowerSets([i for i in range(client_num)])
    sample_sets = random.sample(all_sets, sample_round)
    sample_indices = [power2number(s) for s in sample_sets]
    sv = [[0 for k in range(client_num + 1)] for cid in range(client_num)]
    m = [[0 for k in range(client_num + 1)] for cid in range(client_num)]
    for sample in sample_sets:
        k, index = len(sample), power2number(sample)
        for cid in sample:
            pair = power2number(get_pair(client_num, sample, cid))
            if pair in sample_indices:
                m[cid][k] += 1
                sv[cid][k] += rec_sample[str(index)]["acc"] - rec_sample[str(pair)]["acc"]
    usr_div = lambda x, y: x / y if y != 0 else 0
    shapley_value = [sum(map(usr_div, sv[cid], m[cid])) / client_num for cid in range(client_num)]
    time_cost = sum(rec_sample[str(index)]["time"] for index in sample_indices)
    return shapley_value, 0, time_cost, 0

def MC_Random_SHAP(rec_sample, client_num, sample_round, output_flag=True):
    get_pair = lambda cnum, sample, cid: [c for c in sample if c != cid]
    return SS_SHAP(rec_sample, client_num, sample_round, output_flag, get_pair)

def CC_Random_SHAP(rec_sample, client_num, sample_round, output_flag=True):
    get_pair = lambda cnum, sample, cid: [c for c in range(cnum) if c not in sample]
    return SS_SHAP(rec_sample, client_num, sample_round, output_flag, get_pair)