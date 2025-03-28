# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=3 --train_setup='same';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=3 --train_setup='mixDtr';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=3 --train_setup='mixSize';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=3 --train_setup='noiseX';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=3 --train_setup='noiseY';


# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=5 --train_setup='same';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=5 --train_setup='mixDtr';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=5 --train_setup='mixSize';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=5 --train_setup='noiseX';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=5 --train_setup='noiseY';


# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='same';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='mixDtr';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='mixSize';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='noiseX';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='noiseY';


# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=3 --train_setup='same';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=3 --train_setup='mixDtr';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=3 --train_setup='mixSize';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=3 --train_setup='noiseX';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=3 --train_setup='noiseY';


# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=5 --train_setup='same';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=5 --train_setup='mixDtr';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=5 --train_setup='mixSize';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=5 --train_setup='noiseX';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=5 --train_setup='noiseY';


# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=6 --train_setup='same';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=6 --train_setup='mixDtr';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=6 --train_setup='mixSize';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=6 --train_setup='noiseX';
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='mnist' --client_num=6 --train_setup='noiseY';




# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=6 --train_setup='same';

# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=15 --train_setup='same';


# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=5 --train_setup='same' --all_gpu=4 --now_gpu=1 &
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=5 --train_setup='same' --all_gpu=4 --now_gpu=2 &
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=5 --train_setup='same' --all_gpu=4 --now_gpu=3 &
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=5 --train_setup='same' --all_gpu=4 --now_gpu=4 &
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=15 --train_setup='same' --all_gpu=8 --now_gpu=5;
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=15 --train_setup='same' --all_gpu=8 --now_gpu=6;
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=15 --train_setup='same' --all_gpu=8 --now_gpu=7;
# python fed_driver.py --model='linear_model' --global_round=10 --local_round=4 --dataset='emnist' --client_num=15 --train_setup='same' --all_gpu=8 --now_gpu=8;

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='emnist' --client_num=15 --train_setup='same' --all_gpu=-2 --now_gpu=-1 


# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='same';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixDtr';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixSize';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseX';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseY';

# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='same' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='same' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='same' --all_gpu=3 --now_gpu=3 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='same' --all_gpu=4 --now_gpu=4 &

# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixDtr' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixDtr' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixDtr' --all_gpu=3 --now_gpu=3 &
# python merge_rec_time.py 
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixDtr'

# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixSize' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixSize' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixSize' --all_gpu=3 --now_gpu=3 &

# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='noiseX' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='noiseX' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='noiseX' --all_gpu=3 --now_gpu=3 &

# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='noiseY' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='noiseY' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='cnn_model' --global_round=2 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='noiseY' --all_gpu=3 --now_gpu=3 &






# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='same' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='same' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='same' --all_gpu=3 --now_gpu=3 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='same' --all_gpu=4 --now_gpu=4 &

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixDtr' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixDtr' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixDtr' --all_gpu=3 --now_gpu=3 &
# python merge_rec_time.py 
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixDtr'

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixSize' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixSize' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixSize' --all_gpu=3 --now_gpu=3 &

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseX' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseX' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseX' --all_gpu=3 --now_gpu=3 &

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseY' --all_gpu=3 --now_gpu=1 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseY' --all_gpu=3 --now_gpu=2 &
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseY' --all_gpu=3 --now_gpu=3 &


# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='same';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixDtr';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='mixSize';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='noiseX';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='noiseY';

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='same';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixDtr';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='mixSize';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseX';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=4 --dataset='mnist' --client_num=10 --train_setup='noiseY';

# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='same';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='mixDtr';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='mixSize';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='noiseX';
# python fed_driver.py --model='cnn_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='noiseY';

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=3 --train_setup='same';

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='adult' --client_num=3 --train_setup='same';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='adult' --client_num=6 --train_setup='same';
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='adult' --client_num=10 --train_setup='same';

# python fed_driver.py --model='cnn1d_model' --global_round=4 --local_round=2 --dataset='adult' --client_num=3 --train_setup='same';
# python fed_driver.py --model='cnn1d_model' --global_round=4 --local_round=2 --dataset='adult' --client_num=6 --train_setup='same';
# python fed_driver.py --model='cnn1d_model' --global_round=4 --local_round=2 --dataset='adult' --client_num=10 --train_setup='same';

# python fed_driver.py --model='linear_model' --global_round=2 --local_round=1 --dataset='adult' --client_num=3 --train_setup='same';

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='emnist' --client_num=3 --train_setup='same'
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='emnist' --client_num=6 --train_setup='same'
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='emnist' --client_num=10 --train_setup='same'

# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=3 --train_setup='same'
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=6 --train_setup='same'
# python fed_driver.py --model='linear_model' --global_round=4 --local_round=2 --dataset='mnist' --client_num=10 --train_setup='same'

# python fed_driver.py --model='lstm_model' --global_round=2 --local_round=1 --dataset='sent140' --client_num=10 --train_setup='same'

# python fed_driver.py --model='mlpnlp_model' --global_round=2 --local_round=1 --dataset='sent140' --client_num=3 --train_setup='same'
# python fed_driver.py --model='mlpnlp_model' --global_round=2 --local_round=1 --dataset='sent140' --client_num=6 --train_setup='same'
# python fed_driver.py --model='mlpnlp_model' --global_round=2 --local_round=1 --dataset='sent140' --client_num=10 --train_setup='same'

# CUDA_VISIBLE_DEVICES=3 python sv_main.py

python fed_driver.py --model='lstmtraj_model' --global_round=2 --local_round=1 --dataset='chengdu' --client_num=3 --train_setup='same'
# python fed_driver.py --model='lstmtraj_model' --global_round=2 --local_round=1 --dataset='chengdu' --client_num=6 --train_setup='same'
# python fed_driver.py --model='lstmtraj_model' --global_round=2 --local_round=1 --dataset='chengdu' --client_num=10 --train_setup='same'