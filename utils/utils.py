"""
    Comments:   This is the utility file for running the main program
    ToDo:       * Make it into a class for better usuability
    **********************************************************************************
"""
import numpy as np
import pandas as pd
import os
import json
import shutil
from logging import StreamHandler, INFO, DEBUG, Formatter, FileHandler, getLogger
import pdb
import datetime
def get_KL_list(kl_file, test_sess, KL_thres=40):
    with open(kl_file, 'r') as file:
        kl_divergences_str_keys = json.load(file)
    min_delete = 15
    max_delete = 50
    tmp = {}
    for key in kl_divergences_str_keys:
        # print(key)
        if key.split('-')[0] == test_sess:
            tmp[key] = kl_divergences_str_keys[key]
    sorted_dict_desc = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=True)}
    values_list = list(sorted_dict_desc.values())

    # 检查是否有足够的元素
    if len(values_list) >= min_delete:
        tenth_largest_value = values_list[min_delete - 1]  # 因为索引是从0开始的
    else:
        tenth_largest_value = None
    if len(values_list) > max_delete:
        twenty_largest_value = values_list[max_delete - 1]  # 因为索引是从0开始的
    else:
        twenty_largest_value = None
    # print(sorted_dict_desc)
    sorted_dict_desc = {key: value for key, value in sorted_dict_desc.items() if value > min(KL_thres, tenth_largest_value) or value > max(KL_thres, twenty_largest_value)}
    problemetic_sess = []
    for key in sorted_dict_desc:
        problemetic_sess.append(key.split('-')[1])
    problemetic_sess = ['Dingyi_1' if item == 'Dingyi_2' else 'Dingyi_2' if item == 'Dingyi_3' else item for item in
                     problemetic_sess]

    return problemetic_sess


def get_used_idx(s_names, problematic_sess, sample_per_epoch, val_sess, test_sess):
    bad_idx = []
    all_sess = []
    sep_list = val_sess + test_sess
    for name in s_names:
        if name + '_1' not in sep_list:
            all_sess.append(name + '_1')
        if name + '_2' not in sep_list:
            all_sess.append(name + '_2')
    for sess in problematic_sess:
        if sess in sep_list:
            continue
        sess_idx = all_sess.index(sess)
        start_idx = sess_idx * sample_per_epoch
        bad_idx.extend(range(start_idx, start_idx + sample_per_epoch))

    # 将 bad_idx 转换为集合
    bad_idx_set = set(bad_idx)

    # 使用集合来过滤 use_idx
    use_idx = [idx for idx in range(sample_per_epoch * len(all_sess)) if idx not in bad_idx_set]


    return use_idx


def load_data(mat_path, sess, start=0, end=20000):

    for file in os.listdir(mat_path):

        if file.endswith('.npy'):
            sbj = file.split('_')[0]
            session = file.split('_')[1]
            if sbj + '_' + session == sess:
                # 加载.npy文件
                print(file)
                file_path = os.path.join(mat_path, file)
                print("before load:", datetime.datetime.now().strftime("%H:%M:%S"))
                data = np.load(file_path)[start: end]
                print("after load:", datetime.datetime.now().strftime("%H:%M:%S"))
                lable_path = mat_path.replace('eeg', 'joint')
                label_file = file.replace('EEG', 'Joint')
                label = np.load(os.path.join(lable_path, label_file))[start: end]

                return data, label

def sample_load_data(mat_path ,sess, sample_per_sess):
    for file in os.listdir(mat_path):

        if file.endswith('.npy'):
            sbj = file.split('_')[0]
            session = file.split('_')[1]
            if sbj + '_' + session == sess:
                # 加载.npy文件
                print(file)
                file_path = os.path.join(mat_path, file)
                print("before load:", datetime.datetime.now().strftime("%H:%M:%S"))
                data = np.load(file_path)
                print("after load:", datetime.datetime.now().strftime("%H:%M:%S"))
                lable_path = mat_path.replace('eeg', 'joint')
                label_file = file.replace('EEG', 'Joint')
                label = np.load(os.path.join(lable_path, label_file))
                # 确保每个文件中样本数量足够
                num_samples = data.shape[0]
                if num_samples >= sample_per_sess:
                    # 选择一个随机的起始点
                    start_index = np.random.randint(0, num_samples - 40000)
                    # 从起始点开始选择连续的40000个样本
                    sampled_data = data[start_index:start_index + 40000]
                    sampled_label = label[start_index:start_index + 40000]
                return sampled_data, sampled_label

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()
def dir_maker(path, clean=False):
    """Creating folder structures based on the path specified.

    """

    # First check if the directory exists
    if path.exists():
        print("Path already exists.")
        if clean:
            print("Cleaning")
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    else:
        print("Creating new folders.")
        # Create paths including parent directories
        path.mkdir(parents=True, exist_ok=True)


def set_logger(SAVE_OUTPUT, LOG_FILE_NAME):
    """For better handling logging functionality.

    Obtained and modified from Best practices to log from CS230 Deep Learning, Stanford.
    https://cs230-stanford.github.io/logging-hyperparams.html

    Example:
    ```
    logging.info("Starting training...")
    ```

    Attributes:
        SAVE_OUTPUT: The directory of where you want to save the logs
        LOG_FILE_NAME: The name of the log file

    Returns:
        logger: logger with the settings you specified.
    """

    logger = getLogger()
    logger.setLevel(INFO)

    if not logger.handlers:
        # Define settings for logging
        log_format = Formatter(
            '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
        # for streaming, up to INFO level
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        handler.setFormatter(log_format)
        logger.addHandler(handler)

        # for file, up to DEBUG level
        handler = FileHandler(SAVE_OUTPUT + '/' + LOG_FILE_NAME, 'w')
        handler.setLevel(DEBUG)
        handler.setFormatter(log_format)
        logger.setLevel(DEBUG)
        logger.addHandler(handler)

    return logger


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    return int(hours), int(minutes), seconds


def chunking(x, y, batch_size, future_step, num_chan, num_chan_kin):
    """
    A function to chunk the data into a batch size
    :param x: Input data
    :param y: Output target
    :param future_step: How much further you want to predict
    :return: Chunked matrices both for input and output
    """
    # Initialize the sequence and the next value
    seq, next_val = [], []
    # seq = np.empty(shape=(len(x)-batch_size-future_step, batch_size, future_step))
    # next_val = np.empty(shape=(len(y)+batch_size+future_step-1, num_chan_kin))
    # Based on the batch size and the future step size,
    # run a for loop to create chunks.
    # Here, it's BATCH_SIZE - 1 because we are trying to predict
    # one sample ahead. You could change this to your own way
    # e.g. want to predict 5 samples ahead, then - 5
    for i in range(0, len(x) - batch_size - future_step, future_step):
        seq.append(x[i: i + batch_size, :])
        next_val.append(y[i + batch_size + future_step - 1, :])

    # So now the data is [Samples, Batch size, One step prediction]
    seq = np.reshape(seq, [-1, batch_size, num_chan])
    next_val = np.reshape(next_val, [-1, num_chan_kin])

    X = np.array(seq)
    Y = np.array(next_val)

    return X, Y


def update_args(args, best_params):
    """Update some of the parameters after optuna optimization.

    """

    if args.decode_type in args.rnn_decoders:
        # Regardless of fix_do, init_std is optimized so need to be updated
        args.init_std = float(best_params['init_std'])
        # If layers and hidden units are not fixed, it's optimized so update
        if args.fix_do == 0:
            args.rnn_num_hidden = int(best_params['rnn_num_hidden'])
            args.rnn_num_stack_layers = int(
                best_params['rnn_num_stack_layers'])
    # Do the same for TCN
    elif args.decode_type in args.cnn_decoders:
        args.tcn_num_hidden = int(best_params['tcn_num_hidden'])
        args.tcn_num_layers = int(best_params['tcn_num_layers'])
        args.tcn_kernel_size = int(best_params['tcn_kernel_size'])

    return args

if __name__ == '__main__':
    # 示例调用
    sample_per_epoch = 8000  # 假设每个会话的样本数为8000
    s_names = ['Aung', 'Changhao', 'Chengru', 'Chengxuan', 'David', 'Dingyi',
                   'Donald', 'Dongping', 'Fuxi', 'Hanna', 'Hanwei', 'Hongyu',
                   'Huizhi', 'James', 'Jenny', 'Kairui', 'LiYong', 'Lixun',
                   'Meilun', 'Meiqian', 'Noemie', 'Rosary', 'Rui', 'Ruixuan',
                   'Shangen', 'Shuailei', 'Shuqi', 'Sunhao', 'Tang', 'Wenjin',
                   'Xiaohao', 'Xiaojing', 'Ximing', 'Xueyi', 'Yidan', 'Yiruo',
                   'Youquan', 'Yuan', 'Yueying', 'Yuhao', 'Yunfeng', 'Yuren',
                   'Yuting', 'Zequn', 'Zhangsu', 'Zheren', 'Zhiman', 'Zhisheng',
                   'Zhiwei', 'Zhuoru']
    # problemetic_sess = ['Changhao_1', 'Kairui_1', 'Tang_2', 'Zequn_1', 'Yuan_1', 'Yidan_1', 'Ruixuan_1', 'Meiqian_1', 'Huizhi_2']
    for s_name in s_names:
        # if s_name not in [ 'Huizhi', 'Meiqian', 'Ruixuan']:
        #     continue
        # if s_name < 'Yuting':
        #     continue
        test_sess = s_name + '_2'
        val_sess = s_name + '_1'
        problemetic_sess = get_KL_list('../SbjInd/kl_dict.json', test_sess)
        print(problemetic_sess)
    use_idx = get_used_idx(s_names, problemetic_sess, sample_per_epoch, ['Changhao_1'], ['Aung_2'])
    print(use_idx)