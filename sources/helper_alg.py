from helper_shap import buildPowerSets, power2number
import scipy.special as scp
import numpy as np
import random
import os
import pickle as pk
import time
import math
import copy

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