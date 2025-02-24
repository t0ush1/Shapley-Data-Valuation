import scipy.special as scp
import numpy as np
import random
import math
import copy
from itertools import combinations

random.seed(0)
np.random.seed(0)

import sys

sys.path.append("/code/Shapley-Data-Valuation/sources")

from helper_shap import power2number
from util import Record


def IPSS_SHAP(record: Record, client_num, sample_round, output_flag=True):
    all_clients = [i for i in range(client_num)]
    sample_sets = []
    for i in range(client_num + 1):
        layer_sets = list(map(list, combinations(all_clients, i)))
        layer_num = len(layer_sets)
        if sample_round >= layer_num:
            sample_round -= layer_num
            sample_sets += layer_sets
        else:
            sample_sets += random.sample(layer_sets, sample_round)
            break
    sample_indices = [power2number(s) for s in sample_sets]

    shapley_value_acc = [0 for _ in range(client_num)]
    for cid in range(client_num):
        sv_acc = 0.0
        for subset in sample_sets:
            if not (cid in subset) and (power2number(subset + [cid]) in sample_indices):
                delta_acc = record.get(subset + [cid], "acc") - record.get(subset, "acc")
                sv_acc = sv_acc + delta_acc / scp.comb(client_num - 1, len(subset))
        shapley_value_acc[cid] = sv_acc / client_num

    cost_time = sum([record.get(subset, "time") for subset in sample_sets])
    if output_flag:
        print("IPSS shapley Value ({}):".format(cost_time), "\n", shapley_value_acc)
    return shapley_value_acc, cost_time


def TMC_SHAP(record: Record, client_num, sample_round=30, ouput_flag=True):
    def comp(record: Record, client_num, utils="acc", sample_round=100):
        # all_sets = buildPowerSets(client_num)
        shapley_value_utility = [0 for i in range(client_num)]
        shapley_value_loss = [0 for i in range(client_num)]

        now_perm = [i for i in range(client_num)]
        cost_time_utility = record.get(now_perm, "time")
        # cost_time_loss= rec_sample[str(power2number(now_perm))]['time']

        # hyper_parameters in trunced_monte_carlo_method.
        convergen_rate = 0.001
        perm_delta = 0.0001
        total_round = 0
        convergen_flag = False
        allset_acc = record.get(now_perm, utils)
        temp_utility = [0 for _ in range(client_num + 1)]

        # adopt acc as the utility function
        while (not convergen_flag) and (total_round < sample_round):
            # if total_round% 10==0: print("R{}".format(total_round))
            np.random.shuffle(now_perm)
            total_round = total_round + 1
            # print('Round#{}#: NowPerm is {}'.format(total_round, now_perm))

            last_sv_acc = copy.deepcopy(shapley_value_utility)
            temp_utility[0] = record.get([], utils)
            for j in range(0, client_num):
                sub_perm = now_perm[: j + 1]
                # print("   subperm is {}".format(sub_perm), 'perm_delat is {}'.format(perm_delta))
                if np.abs(allset_acc - temp_utility[j]) < perm_delta:
                    temp_utility[j + 1] = temp_utility[j]
                else:
                    temp_utility[j + 1] = record.get(sub_perm, utils)
                    cost_time_utility += record.get(sub_perm, "time")

                # print("\tSubPerm is {}, Update SubPerm[j]={}".format(sub_perm, sub_perm[j]), )
                shapley_value_utility[sub_perm[j]] = (total_round - 1) / total_round * shapley_value_utility[
                    sub_perm[j]
                ] + (1 / total_round) * (temp_utility[j + 1] - temp_utility[j])
            now_coverage = [abs(shapley_value_utility[_] - last_sv_acc[_]) for _ in range(len(last_sv_acc))]
            if sum(now_coverage) < convergen_rate or total_round > sample_round:
                # print("({}/{})".format(total_round, sample_round))
                convergen_flag = True

        if utils == "loss":
            shapley_value_utility = [-1 * sv for sv in shapley_value_utility]
        return shapley_value_utility, cost_time_utility

    # print("Now run the TMC_SHAP. with output={}".format(ouput_flag))

    sample_round = sample_round // client_num
    shapley_value_acc, cost_time_acc = comp(record, client_num, "acc", min(2**client_num, sample_round))

    if ouput_flag:
        print("TMC shapley Value ({}):".format(cost_time_acc), "\n", shapley_value_acc)

    return shapley_value_acc, cost_time_acc


def GTB_SHAP(record: Record, client_num, sample_round=100, ouput_flag=True):
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
                constraints.append(x[i] >= 0)
                for j in range(n):
                    constraints.append(cp.abs(x[i] - x[j] - U[i][j]) <= exps)
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
        exps = 1 / (2 * math.sqrt(U.shape[0]))
        res = []
        while (len(res) == 0) and (exps < 10):
            res = solve_feasibility(U, u_all, exps)
            exps = exps * 1.1

        if len(res) == 0:
            return [0 for _ in range(U.shape[0])]
        return res

    # now_perm = [i for i in range(client_num)]
    def comp(record: Record, client_num, utils, sample_round=100):
        time_cost = 0
        Z = 2 * sum([1 / k + 1 / (client_num - k) for k in range(1, client_num - 1)])
        q_values = [1 / Z * (1 / k + 1 / (client_num - k)) for k in range(1, client_num)]
        k_values = np.arange(1, client_num)
        U = np.zeros([client_num, client_num], dtype=np.float32)
        U_t = [0 for _ in range(sample_round)]

        total_round = 0
        beta = [[0 for _ in range(client_num)] for __ in range(sample_round + 1)]

        while total_round < sample_round:
            round_k = np.random.choice(k_values, size=100, p=q_values / np.sum(q_values))[0]
            k_t = 2
            random_subset = []
            for j in range(k_t):
                new_subset = random.sample(range(0, client_num), round_k)
                for cid in random_subset:
                    beta[total_round][cid] = 1
                random_subset = list(set(random_subset + new_subset))

            if utils == "loss":
                U_t[total_round] = -record.get(random_subset, utils)
            else:
                U_t[total_round] = record.get(random_subset, utils)
            time_cost += record.get(random_subset, "time")
            total_round += 1

        # print("U_t is {}, Z is {}, time_cost is {}".format(U_t, Z, time_cost))
        for i in range(0, client_num):
            for j in range(i, client_num):
                U[i][j] = Z / sample_round * sum([U_t[t] * (beta[t][i] - beta[t][j]) for t in range(total_round)])

        allset = [i for i in range(0, client_num)]
        U_D = record.get(allset, utils)
        time_cost += record.get(allset, "time")
        shapley_value = list(solveF(U, U_D))

        return shapley_value, time_cost

    # shapley_value_acc, cost_time_acc = comp(rec_sample, client_num, 'acc', min(sample_round, 2**client_num))
    # shapley_value_loss, cost_time_loss = comp(rec_sample, client_num, 'loss', min(sample_round, 2**client_num))

    shapley_value_acc, cost_time_acc = comp(record, client_num, "acc", min(sample_round, 2**client_num))
    
    if ouput_flag:
        print("GTB shapley Value ({}):".format(cost_time_acc), "\n", shapley_value_acc)

    return shapley_value_acc, cost_time_acc


def LOO_SHAP(record: Record, client_num, sample_round, output_flag=True):

    def comp(record: Record, client_num, utils):

        allset = [i for i in range(client_num)]
        # time_cost = 0
        allset_utility = record.get(allset, utils)
        time_cost = record.get(allset, "time")

        loo_value = [0 for i in range(client_num)]
        for cid in range(client_num):
            subset = list(set(allset) - set([cid]))
            loo_value[cid] = allset_utility - record.get(subset, utils)
            time_cost += record.get(subset, "time")
        if utils == "loss":
            loo_value = [-sv for sv in loo_value]
        return loo_value, time_cost

    shapley_value_acc, cost_time_acc = comp(record, client_num, "acc")
    if output_flag:
        print("LOO shapley Value ({}):".format(cost_time_acc), "\n", shapley_value_acc)

    return shapley_value_acc, cost_time_acc


def CC_SHAP(record: Record, client_num, sample_round=150, output_flag=True):
    # compute the shapley value by complementary contributions
    def comp(record: Record, client_num, utils="acc", sample_round=100):
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

            time_cost = time_cost + record.get(l_subset, "time") + record.get(r_subset, "time")

            delta_utility = record.get(l_subset, utils) - record.get(r_subset, utils)

            for cid in l_subset:
                temp_rsv[cid][num] += delta_utility
                temp_cnt[cid][num] += 1

            for cid in r_subset:
                temp_rsv[cid][client_num - num] -= delta_utility
                temp_cnt[cid][client_num - num] += 1

        usr_div = lambda x, y: 0 if abs(y) < 1e-8 else x / y
        for cid in range(client_num):
            shapley_value[cid] = 1 / client_num * sum(map(usr_div, temp_rsv[cid], temp_cnt[cid]))

        if utils == "loss":
            shapley_value = [-sv for sv in shapley_value]
        return shapley_value, time_cost

    shapley_value_acc, cost_time_acc = comp(record, client_num, "acc", min(sample_round, 2**client_num))
    if output_flag:
        print("CC shapley Value ({}):".format(cost_time_acc), "\n", shapley_value_acc)

    return shapley_value_acc, cost_time_acc
    # pass
