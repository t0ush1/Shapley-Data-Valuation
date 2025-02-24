import tensorflow as tf
import tensorflow_federated as tff

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    # tf.config.experimental.set_virtual_device_configuration(device,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=200)])
    tf.config.experimental.set_memory_growth(device, True)

import argparse as ap  
import numpy as np
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pickle as pk
import random as rd
rd.seed(42)


exist_dataset_names = ["mnist", "emnist", "gldv2", "stackoverflow", "cifa100", "shakespeare"]

def create_dataset(name, c_num, c_type):
    if not (name in exist_dataset_names):
        print(name + " is not in " + str(exist_dataset_names) + "!")
        return
    
    if name == 'emnist':
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()  

        # create train_dataset for cnum clients.
        ids = emnist_train.client_ids
        client_ids = [ids[i:i+len(ids)//c_num] for i in range(0, len(ids), len(ids)//c_num)]
        client_data = [[] for i in range(c_num)] 
        for i in range(c_num):
            for id in client_ids[i]:
                client_data[i] += list(emnist_train.create_tf_dataset_for_client(id))
        
        client_trainX, client_trainY = [[] for i in range(c_num)], [[] for i in range(c_num)]
        for i in range(c_num):
            client_trainX[i] = np.array([data_item['pixels'] for data_item in client_data[i]])
            client_trainY[i] = np.array([data_item['label']  for data_item in client_data[i]])
               
        # create test_dataset for all_users.
        test_data = list(emnist_test.create_tf_dataset_from_all_clients())
        testX = np.array([data_item['pixels'] for data_item in test_data])
        testY = np.array([data_item['label'] for data_item in test_data])

        # save the dataset as type of tf.keras.dataset.
        if not os.path.exists('./emnist/'): os.mkdir('./emnist/')
        data_path  = "./emnist/client_"+str(c_num)+"_"+str(c_type)+'/'
        if not os.path.exists(data_path): os.mkdir(data_path)
        for cid in range(c_num):
            with open(data_path+"client_trainX_"+str(cid)+".pk", 'wb')  as fout:
                pk.dump(client_trainX[cid], fout)
            with open(data_path+"client_trainY_"+str(cid)+".pk", 'wb')  as fout:
                pk.dump(client_trainY[cid], fout)

        with open(data_path+"testX.pk", 'wb')  as fout:
            pk.dump(testX, fout)        
        with open(data_path+"testY.pk", 'wb')  as fout:
            pk.dump(testY, fout)
        # print("Create" + str(c_type) + "-type dataset for " +str(c_num)+ " clients over " + str(name))

    if name == 'mnist':
        (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
        # train_images, test_images = train_images / 255.0, test_images / 255.0
        trainX, testX = trainX / 255.0, testX / 255.0

        min_samples = min(np.bincount(trainY))
        trainXbyL = [trainX[np.random.choice(np.where(trainY == _)[0], min_samples, replace=False)] for _ in range(10)]
        trainYbyL = [trainY[np.random.choice(np.where(trainY == _)[0], min_samples, replace=False)] for _ in range(10)]
        client_trainX, client_trainY = [[] for _ in range(c_num)], [[] for _ in range(c_num)]


        if c_type == 'same':
            # client_ids = []
            for cid in range(c_num):
                for yid in range(10):
                    client_trainX[cid].extend(trainXbyL[yid][cid*min_samples//c_num:(cid+1)*min_samples//c_num])
                    client_trainY[cid].extend(trainYbyL[yid][cid*min_samples//c_num:(cid+1)*min_samples//c_num])

        if c_type == 'mixDtr':
            # create a Dominated dataset and an Average Dataset for all clients.
            # Each client gets 10*0.8 MinSample // cum  from Domninated Dataset.
            # Each client gets 10*0.2 MinSample // cum  from Average Dataset.
            dm_ratio = 0.8
            dm_num   = int(dm_ratio*min_samples)
            av_num   = int((1-dm_ratio)*min_samples)

            DomClientX, DomClientY = [], []

            for yid in range(10): 
                DomClientX.extend(trainXbyL[yid][:dm_num])
                DomClientY.extend(trainYbyL[yid][:dm_num])
            
            # Get the data from dominated dataset.
            for cid in range(c_num):
                client_trainX[cid].extend(DomClientX[10*cid*dm_num//c_num: 10*(cid+1)*dm_num//c_num])
                client_trainY[cid].extend(DomClientY[10*cid*dm_num//c_num: 10*(cid+1)*dm_num//c_num])
                
                for yid in range(10):
                    client_trainX[cid].extend(trainXbyL[yid][dm_num+cid*av_num//c_num: dm_num+(cid+1)*av_num//c_num])
                    client_trainY[cid].extend(trainYbyL[yid][dm_num+cid*av_num//c_num: dm_num+(cid+1)*av_num//c_num])

        if c_type == 'mixSize':
            tot = c_num * (c_num + 1) // 2
            cid_size = [int(min_samples * (i + 1) / tot) for i in range(c_num)]
            tot_size = [sum(cid_size[:i]) for i in range(c_num + 1)]

            for cid in range(c_num):
                for yid in range(10):
                    client_trainX[cid].extend(trainXbyL[yid][tot_size[cid]:tot_size[cid+1]])
                    client_trainY[cid].extend(trainYbyL[yid][tot_size[cid]:tot_size[cid+1]])

        if c_type == 'noiseX':

            for cid in range(c_num):
                for yid in range(10):
                    client_trainX[cid].extend(trainXbyL[yid][cid*min_samples//c_num:(cid+1)*min_samples//c_num])
                    client_trainY[cid].extend(trainYbyL[yid][cid*min_samples//c_num:(cid+1)*min_samples//c_num])

            # Add noise ~ N(0, noise[cid]) to c_train_x[cid]
            max_noise = 0.2
            noise = [max_noise/(cid+1) for cid in range(c_num)]
            for cid in range(c_num):
                client_trainX[cid] = np.array(client_trainX[cid])
                noise_to_add = np.random.normal(0, 1, client_trainX[cid].shape)
                client_trainX[cid] += noise_to_add * noise[cid]
                client_trainX[cid] = np.clip(client_trainX[cid], 0, 1)
        

        if c_type == 'noiseY':

            for cid in range(c_num):
                for yid in range(10):
                    client_trainX[cid].extend(trainXbyL[yid][cid*min_samples//c_num:(cid+1)*min_samples//c_num])
                    client_trainY[cid].extend(trainYbyL[yid][cid*min_samples//c_num:(cid+1)*min_samples//c_num])

            # Replace the label of c_train_y[cid] by noisey[cid]
            max_noise = 0.2
            noisey = [max_noise/(cid+1) for cid in range(c_num)]
            for cid in range(c_num):
                client_trainY[cid] = np.array(client_trainY[cid])
                num_labels_to_replace = int(len(client_trainY[cid]) * noisey[cid])
                indices_to_replace = np.random.choice(len(client_trainY[cid]), num_labels_to_replace, replace=False)
                for idx in indices_to_replace:
                    client_trainY[cid][idx] = np.random.randint(0, 10) 

        for cid in range(c_num):
            # rd.shuffle(client_trainX[cid])
            # rd.shuffle(client_trainY[cid])
            client_trainX[cid] = np.array(client_trainX[cid])
            client_trainY[cid] = np.array(client_trainY[cid])
            indices = list(range(len(client_trainX[cid])))
            rd.Random(42).shuffle(indices)
            client_trainX[cid] = np.array([client_trainX[cid][i] for i in indices])
            client_trainY[cid] = np.array([client_trainY[cid][i] for i in indices])



        testX = np.array(testX)
        testY = np.array(testY)
        # save the dataset as type of tf.keras.dataset.
        if not os.path.exists('./mnist/'): os.mkdir('./mnist/')
        data_path  = "./mnist/client_"+str(c_num)+"_"+str(c_type)+'/'
        if not os.path.exists(data_path): os.mkdir(data_path)
        for cid in range(c_num):
            # print("Now we create (size={}):".format(len(client_trainX[cid])), data_path+"client_trainX_"+str(cid)+".pk")
            with open(data_path+"client_trainX_"+str(cid)+".pk", 'wb')  as fout:
                pk.dump(client_trainX[cid], fout)
            with open(data_path+"client_trainY_"+str(cid)+".pk", 'wb')  as fout:
                pk.dump(client_trainY[cid], fout)

        with open(data_path+"testX.pk", 'wb')  as fout:
            pk.dump(testX, fout)        
        with open(data_path+"testY.pk", 'wb')  as fout:
            pk.dump(testY, fout)


    


    print("Create " + str(c_type) + "-type dataset for " +str(c_num)+ " clients over " + str(name))
            # pass
        
if __name__ == "__main__":

    #  example: python fed_data_creater.py --dataset='emnist' --c_num=10 --c_type='same'  
    #  example: python fed_data_creater.py --dataset='mnist'  --c_num=10 --c_type='same', 'mixDtr' , 'mixSize', 'noiseX', 'noiseY'
    parser = ap.ArgumentParser(description="Create Federated Dataset.")
    parser.add_argument('--dataset', type=str, default='emnist')
    parser.add_argument('--c_num', type=int, default=3)
    parser.add_argument('--c_type', type=str, default='same')
    args = parser.parse_args()
    create_dataset(args.dataset, args.c_num, args.c_type)

