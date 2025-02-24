from collections import defaultdict
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def load_dataset():
    data_path = "./sent140/training.1600000.processed.noemoticon.csv"
    data = pd.read_csv(data_path, encoding="latin-1", header=None)
    print(data[0].value_counts())
    exit(0)
    labels = data[0].astype("category").cat.codes
    users = data[4]
    texts = data[5]

    tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    print(len(word_index))
    print(list(word_index.items())[:10])

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=30, padding="post", truncating="post")
    print(padded_sequences[:5])

    user_data = defaultdict(lambda: ([], []))
    for user, sequence, label in zip(users, padded_sequences, labels):
        user_data[user][0].append(sequence)
        user_data[user][1].append(label)
    print(len(user_data))

    return user_data


def dump_data(dir, filename, data, indices, ratio=0.1):
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

    data_path = f"./sent140/client_{cnum}_same"
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    X_test, y_test = [], []
    for user in test_users:
        X_test.extend(user_data[user][0])
        y_test.extend(user_data[user][1])
    indices = np.random.permutation(len(y_test))
    dump_data(data_path, "testX.pk", X_test, indices)
    dump_data(data_path, "testY.pk", y_test, indices)
    print("test:", len(test_users), len(y_test), *np.bincount(y_test))

    for cid in range(cnum):
        X_client, y_client = [], []
        for user in client_users[cid]:
            X_client.extend(user_data[user][0])
            y_client.extend(user_data[user][1])
        indices = np.random.permutation(len(y_client))
        dump_data(data_path, f"client_trainX_{cid}.pk", X_client, indices)
        dump_data(data_path, f"client_trainY_{cid}.pk", y_client, indices)
        print(f"client {cid}:", len(client_users[cid]), len(y_client), *np.bincount(y_client))


if __name__ == "__main__":
    user_data = load_dataset()
    for cnum in [3, 6, 10]:
        create_dataset(cnum, user_data)
