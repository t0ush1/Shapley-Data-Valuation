from collections import defaultdict
import os
import pickle
import numpy as np
import pandas as pd


def load_dataset(input_len=10, output_len=1):
    data_path = "./chengdu/gps_20161101"
    data = pd.read_csv(data_path, names=["driver_id", "order_id", "timestamp", "lon", "lat"], nrows=500000)
    print(data["driver_id"].nunique(), data["order_id"].nunique())

    user_data = defaultdict(lambda: ([], []))

    for (driver_id, order_id), group in data.groupby(["driver_id", "order_id"]):
        group = group.sort_values(by="timestamp")
        traj = group[["lon", "lat"]].values

        traj_len = len(traj)
        if traj_len < input_len + output_len:
            continue

        input_list, output_list = [], []
        for i in range(traj_len - input_len - output_len + 1):
            input_list.append(traj[i : i + input_len])
            output_list.append(traj[i + input_len : i + input_len + output_len])

        user_data[driver_id][0].extend(input_list)
        user_data[driver_id][1].extend(output_list)

    return user_data


def dump_data(dir, filename, data, indices, ratio=1):
    data = np.array(data)[indices]
    data = data[:int(len(data) * ratio)]
    with open(os.path.join(dir, filename), "wb") as fout:
        pickle.dump(data, fout)


def create_dataset(cnum, user_data, test_size=0.2):
    users = list(user_data.keys())
    np.random.shuffle(users)
    test_users = users[:int(len(users) * test_size)]
    train_users = users[int(len(users) * test_size):]
    unum = len(train_users) // cnum
    client_users = [train_users[cid * unum : (cid + 1) * unum] for cid in range(cnum)]

    data_path = f"./chengdu/client_{cnum}_same"
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    X_test, y_test = [], []
    for user in test_users:
        X_test.extend(user_data[user][0])
        y_test.extend(user_data[user][1])
    indices = np.random.permutation(len(y_test))
    dump_data(data_path, "testX.pk", X_test, indices)
    dump_data(data_path, "testY.pk", y_test, indices)
    print("test:", len(test_users), len(y_test))

    for cid in range(cnum):
        X_client, y_client = [], []
        for user in client_users[cid]:
            X_client.extend(user_data[user][0])
            y_client.extend(user_data[user][1])
        indices = np.random.permutation(len(y_client))
        dump_data(data_path, f"client_trainX_{cid}.pk", X_client, indices)
        dump_data(data_path, f"client_trainY_{cid}.pk", y_client, indices)
        print(f"client {cid}:", len(client_users[cid]), len(y_client))


if __name__ == "__main__":
    user_data = load_dataset()
    for cnum in [3, 6, 10]:
        create_dataset(cnum, user_data)
