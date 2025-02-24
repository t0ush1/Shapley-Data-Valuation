import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow as tf
import numpy as np
import gc

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    # tf.config.experimental.set_virtual_device_configuration(device,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=200)])
    tf.config.experimental.set_memory_growth(device, True)


CLINT_NUM = 5
BASIC_PORT = 50100
STOP_PORT =  60100
PORT_GAP = 25
FED_ROUND = 10
# DATA_SHAPE = 
LOCAL_EPOCH = 5
LOCAL_BATCH = 64


FED_SHAPE_DICT = {"emnist":((28,28), 10), "mnist":((28,28), 10), "celeba":((84,84,3), 2), "adult":((14,), 2), "adult_onehot":((105,), 2), "covtype": ((54,), 7), "census": ((41,), 2), "sent140": ((30,), 2)}


def nparray_to_rpcio(nparray):
    byte_array_data = [x.tobytes() for x in nparray]
    byte_array_type = [str(x.dtype) for x in nparray]
    byte_array_shape = [str(x.shape) for x in nparray]
    return byte_array_data, byte_array_type, byte_array_shape 

def rpcio_to_nparray(byte_data, byte_type, byte_shape):

    return [np.frombuffer(data, dtype=np.dtype(rtype)).reshape(eval(shape)) 
            for data,rtype,shape in zip(byte_data, byte_type, byte_shape)]


class basic_model():

    def __init__(self, _input, _output, _type):

        # self.local_mini_flag = LOCAL_MINI_FLAG
        # self.local_mini_batch_size = LOCAL_MINI_BATCH_SIZE
        # self.local_mini_epoch = LOCAL_MINI_EPOCH
        
        # print("Local_Mini_Batch is {}, with Mini_Size is {}".format(self.local_mini_flag, self.local_mini_batch_size))
        # pass
        # self.input  = _input
        # # self.output = _output

        print("Now we are creating {} with input={}, output={}".format(_type, _input,_output))


    
    def model_compile(self):
        self.model.compile(optimizer="adam", 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    def model_fit(self, _datasets, _local_epoches, _batchsize):
        # if not self.local_mini_flag:
        if len(_datasets[0]):
            self.model.fit(_datasets[0], _datasets[1], batch_size=_batchsize, epochs=_local_epoches, verbose=1)
        # else:
        #     ids = np.random.choice(np.array(range(len(_datasets[1]))),size=self.local_mini_batch_size, replace=False)
        #     mini_datasets = (_datasets[0][ids], _datasets[1][ids])
        #     # print("ids:", ids, '\n', "labels:", mini_datasets[1])
        #     self.model.fit(mini_datasets[0], mini_datasets[1], batch_size=self.local_mini_batch_size, epochs=self.local_mini_epoch)


    def model_load_weights(self, _weights):
        self.model.set_weights(_weights)


    def model_get_weights(self):
        return self.model.get_weights()
    
    def model_get_eval(self, _test_data, _test_label, notes="",verbose=2):
        # print(notes, end='\t')
        return self.model.evaluate(_test_data, _test_label, verbose=1)
    
    def model_save(self, cid):
        self.model.save(f"./scalability/model/model_{cid}")
        del self.model
        gc.collect()
        tf.keras.backend.clear_session()
    
    def model_load(self, cid):
        self.model = tf.keras.models.load_model(f"./scalability/model/model_{cid}")


class linear_model(basic_model):
    # 0：构建模型，随机初始化参数，1：构建模型，加载参数，2：不构建模型
    def  __init__(self, _input, _output, init_mode=0):
        self.input = _input
        self.output = _output
        self.model_type = "Linear Model"
        super().__init__(self.input, self.output, self.model_type)
        assert init_mode in [0, 1, 2]
        if init_mode == 2:
            self.model = None
        else:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=self.input),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.output)
            ])
            self.model_compile()
            if init_mode == 1:
                import pickle
                with open("./scalability/model/init_weights", "rb") as f:
                    self.model_load_weights(pickle.load(f))


class cnn_model(basic_model):
    
    def  __init__(self, _input, _output):
        self.input = _input
        self.output = _output
        self.model_type = "CNN Model"
        super(cnn_model, self).__init__(self.input, self.output, self.model_type)

        # Last Used Model
        # self.model = tf.keras.models.Sequential([
        #     tf.keras.layers.Reshape((self.input[0], self.input[1],1), input_shape=self.input),
        #     tf.keras.layers.Conv2D(32, (8,8), activation='relu', padding='same', name='Conv2D-3x3'),  #(28, 28, 32)
        #     tf.keras.layers.MaxPooling2D((8,8), name='Pool2D-2x2'),   # (14,14,32)
        #     tf.keras.layers.Flatten(), #5*5*64
        #     tf.keras.layers.Dense(16, activation='relu'),
        #     tf.keras.layers.Dense(self.output,)
        # ])

        # self.model = tf.keras.models.Sequential([
        #     tf.keras.layers.Reshape((28,28,1), input_shape=(28, 28)),
        #     tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', name='Conv2D-3x3'),  
        #     tf.keras.layers.MaxPooling2D((2,2), name='Pool2D-2x2'),  
        #     tf.keras.layers.Conv2D(64, (2,2),padding='same', activation='relu'), 
        #     tf.keras.layers.MaxPooling2D((2,2)), 
        #     tf.keras.layers.Flatten(), 
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(10,)
        # ])

        # copy from github ...
        self.model = tf.keras.Sequential([
            # Reshape input to match the expected shape
            tf.keras.layers.Reshape((self.input[0], self.input[1], 1), input_shape=self.input),

            # tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
            
            # First convolutional layer
            tf.keras.layers.Conv2D(32, kernel_size=(8, 8), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            
            # Second convolutional layer
            tf.keras.layers.Conv2D(64, kernel_size=(4, 4), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            
            # Flatten and output layer
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.output)
        ])
    

        self.model_compile() 


class cnnrgb_model(basic_model):
    def  __init__(self, _input, _output):
        self.input = _input
        self.output = _output
        self.model_type = "CNNRGB Model"
        super().__init__(self.input, self.output, self.model_type)
        self.model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=_input),
            tf.keras.layers.Conv2D(32, kernel_size=(8, 8), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            tf.keras.layers.Conv2D(64, kernel_size=(4, 4), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.output)
        ])
        self.model_compile()


class cnn1d_model(basic_model):
    def  __init__(self, _input, _output):
        self.input = _input
        self.output = _output
        self.model_type = "CNN1D Model"
        super().__init__(self.input, self.output, self.model_type)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Reshape((self.input[0], 1), input_shape=self.input),
            tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.output)
        ])
        self.model_compile()


class lstm_model(basic_model):
    def  __init__(self, _input, _output):
        self.input = _input
        self.output = _output
        self.model_type = "LSTM Model"
        super().__init__(self.input, self.output, self.model_type)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64, mask_zero=True, input_shape=_input),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(self.output)
        ])
        self.model_compile()


class mlpnlpmodel(basic_model):
    def  __init__(self, _input, _output):
        self.input = _input
        self.output = _output
        self.model_type = "MLP NLP Model"
        super().__init__(self.input, self.output, self.model_type)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5000, output_dim=64, mask_zero=True, input_shape=_input),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.output)
        ])
        self.model_compile()


FED_MODEL_DICT = {"cnn_model":cnn_model, "linear_model":linear_model, "cnnrgb_model":cnnrgb_model, "cnn1d_model":cnn1d_model, "lstm_model":lstm_model, "mlpnlp_model":mlpnlpmodel}
