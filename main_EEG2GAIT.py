import os
import numpy as np
import torch
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from gppdataset_new import GPPDataSet
from utils.utils import params_count
from utils.ymlfun import get_args, get_parser
from models.Gaitnet_model import GaitNet, GaitNet_noAT
from models.GaitGraph_model import GaitGraph, GaitCNN, GaitGraph_tmp
from models.EEGGaitTransformer import EEGGaitTransformer
from models.DCN_model import deepConvNet
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from utils.myaml import load_config
from tqdm import tqdm
import time
from utils.FreDF import FreDFLoss, FreDFLossWithEncouraging, MSEWithEncouragingLoss, GradientNormalizationLoss, FreqGradientLoss
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler


def calc_rval(ind_act, ind_pred):
    '''Calculate r-value and store them in a dict for each joint

    Arguments:
        ind_act:    a list of actual joint angle values [num samples x num joints]
        ind_pred:   a list of predicted joint angles [num samples x num joints]


    Returns:
        r_vals:     a dictionary containing r-value for each joint
    '''
    r_vals = 0
    if (isinstance(ind_act, np.ndarray)):
        ind_act = torch.from_numpy(ind_act)
    if (isinstance(ind_pred, np.ndarray)):
        ind_pred = torch.from_numpy(ind_pred)
    ind_act = ind_act.to(device)
    ind_pred = ind_pred.to(device)
    tmp = []
    for i in range(ind_act.shape[1]):
        x = ind_act[:, i]
        y = ind_pred[:, i]
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        r_vals += cost
        tmp.append(float(cost))
    print(tmp)
    return r_vals / ind_act.shape[1]


def evaluate(model, testdata, testTarget, domain=None, scJoints=None):
    print('start evaluate')
    model.eval()
    with torch.no_grad():
        batch_size = 100
        for i in range(int(len(testdata) / batch_size)):
            s = i * batch_size
            e = i * batch_size + batch_size

            inputs = torch.tensor(testdata[s:e]).to(device)
            target = torch.tensor(testTarget[s:e]).to(device)

            inputs = Variable(inputs)
            target = Variable(target)
            # zero the parameter gradients
            outputs, contrastive_features, alignment_loss = model(inputs)
            if (i == 0):
                testPreJoints = outputs.cpu().detach().numpy()
            else:
                testPreJoints = np.vstack([testPreJoints, outputs.cpu().detach().numpy()])

        if (e != len(testdata)):
            inputs = torch.tensor(testdata[e:]).to(device)
            inputs = Variable(inputs)
            target = torch.tensor(testTarget[e:]).to(device)
            target = Variable(target)
            outputs, contrastive_features, alignment_loss = model(inputs)
            testPreJoints = np.vstack([testPreJoints, outputs.cpu().detach().numpy()])
        rval = calc_rval(testTarget[:, [i for i in used_joints]], testPreJoints[:, [i for i in used_joints]])
        r2score = r2_score(testTarget[:, [i for i in used_joints]], testPreJoints[:, [i for i in used_joints]])
    model.train()

    return rval, r2score


def run(time_step):

    if not os.path.exists('./checkpoints/'.format(model_name)):
        os.mkdir('./checkpoints/'.format(model_name))
   
    count_flag = 0
    for p_id in range(0,1):

        for trial in range(2, 3):
            if model_name.startswith('EEG2Gait'):
                from models.EEG2Gait import EEG2Gait
                model = EEG2Gait(config).to(device)
            if (count_flag == 0):
                num_param = params_count(model)
                count_flag = 1
            print('number of param of model:', num_param)
            batch_size = 100
            if loss_name == 'freq_mse':
                criterion = FreDFLoss()
            elif loss_name == 'reward_mse':
                criterion = MSEWithEncouragingLoss()
            elif loss_name == 'freq_reward_mse':
                criterion = FreDFLossWithEncouraging()
            elif loss_name == 'mse':
                criterion = MSELoss()
            elif loss_name == 'freq_grad':
                criterion = FreqGradientLoss()
            elif loss_name == 'grad':
                criterion = GradientNormalizationLoss()
            optimizer = optim.Adam(model.parameters())
            print(mat_path)
            config.eegnet.val_trial = 20
            dataset = GPPDataSet(mat_path, time_step, val_trial=config.eegnet.val_trial, test_trial=config.eegnet.test_trial, start_trial=0)
            s_dic = dataset.get_data(laplacian=1, remove_EMG=peri_remove)

            trainEEG, valEEG, testEEG = s_dic['trainEEG'], s_dic['valEEG'], s_dic['testEEG']

            trainTarget, valTarget, testTarget = s_dic['trainTarget'], s_dic['valTarget'], s_dic['testTarget']

            scJoints = s_dic['scJoints']

            trainEEG = np.transpose(trainEEG, (0, 2, 1))  # 70000 * 1 * 60 * T
            valEEG = np.transpose(valEEG, (0, 2, 1))
            testEEG = np.transpose(testEEG, (0, 2, 1))
            trainEEG = np.expand_dims(trainEEG, 1)
            valEEG = np.expand_dims(valEEG, 1)
            testEEG = np.expand_dims(testEEG, 1)

            X_train = trainEEG.astype('float32')
            y_train = trainTarget.astype('float32')

            X_val = valEEG.astype('float32')
            y_val = valTarget.astype('float32')
            X_test = testEEG.astype('float32')
            y_test = testTarget.astype('float32')


            print('get_data ')
            print('shape of training set = ', X_train.shape)
            print('shape of test set = ', X_test.shape)

            highest_val = 0
            flag = 0
            config.num_teacher_epoch = 30
            model.train()
            for epoch in range(config.num_teacher_epoch):  # loop over the dataset multiple times
                flag += 1
                if not os.path.exists('./checkpoints/' + model_name):
                    os.mkdir('./checkpoints/' + model_name)
                print("\nEpoch ", epoch)

                running_loss = 0.0
                print('start training', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                for i in tqdm(range(int(len(X_train) / batch_size))):
                    s = i * batch_size
                    e = i * batch_size + batch_size

                    inputs = torch.from_numpy(X_train[s:e]).to(device)  # (batch_size * 1 * channel * T)

                    labels = torch.FloatTensor(np.array([y_train[s:e]]) * 1.0).to(device)
                    labels = torch.squeeze(labels)
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs, contrastive_features, alignment_loss = model(inputs)
                    if i == 0:
                        testPreJoints = outputs.cpu().detach().numpy()
                    else:
                        testPreJoints = np.vstack([testPreJoints, outputs.cpu().detach().numpy()])
                    
                    loss = criterion(outputs, labels)
                    
                    total_loss = loss
                    total_loss.backward()

                    optimizer.step()
                    # print(loss)

                    running_loss += loss.data
                if (e != len(X_train)):
                    inputs = torch.tensor(X_train[e:]).to(device)
                    labels = torch.FloatTensor(np.array([y_train[e:]]) * 1.0).to(device)
                    labels = torch.squeeze(labels)
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)

                    outputs, contrastive_features, alignment_loss = model(inputs)#, labels)
                    testPreJoints = np.vstack([testPreJoints, outputs.cpu().detach().numpy()])

                # Validation accuracy
                print('finish training', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                train_rval = calc_rval(y_train[:, -6:], testPreJoints)


                train_r2score = r2_score(y_train[:, -6:], testPreJoints)
                val_rval, val_r2score = evaluate(model, X_val, y_val[:, -6:])
                test_rval, test_r2score = evaluate(model, X_test, y_test[:, -6:])

                # if (val_rval > highest_val):
                if (test_rval > highest_val):
                    save_path = result_base + '/{}_{}_{}_{}{}_TS{}_{}_'.format(channel_select, sbj_name, session, config.eegnet.val_trial, config.eegnet.test_trial, str(config.eegnet.eeg.time_step), band)

                    save_path = save_path + str(float(test_rval) * 100)[:5] + '.pt'

                    best_model_state = model.state_dict()
                    flag = 0

                    highest_val = val_rval
                    best_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss,

                        'train_rval': train_rval,
                        'train_r2score': train_r2score,
                        'val_rval': val_rval,
                        'val_r2score': val_r2score,
                        'test_rval': test_rval,
                        'test_r2score': test_r2score
                    }

                if (flag > config.patience):
                    break
                elif (epoch > 20):

                    model.load_state_dict(best_dict['model_state_dict'])

                print("Training Loss ", running_loss)
                print("Train PCC = {}, Train r2_score = {}".format(train_rval, train_r2score))
                print("val_ PCC = {}, val r2_score = {}".format(val_rval, val_r2score))
                print("Test PCC = {}, Test_r2_score = {}".format(test_rval, test_r2score))
           
        torch.save(best_dict, save_path)


def check(name, session, path):
    if (name in path and session in path):
        return True
    return False



if __name__ == '__main__':
    # base = 'D:/Datasets/GPP/data/21Jun Experiment - Jenny (sess-1)/eeg/walking_block1/'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Channel names
    ch_names = [
        'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
        'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2',
        'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
        'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
        'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4',
        'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2'
    ]

    peri_remove = 0
    # Neighbors as a dictionary
    if not True:
        channel_neighbors = {"Fp1": [],
                                    "Fp2": [],
                                    "AF7": [],
                                    "AF3": [],
                                    "AFz": [],
                                    "AF4": [],
                                    "AF8": [],
                                    "F7": [],
                                    "F5": [],
                                    "F3": ['FC3', 'F1'],
                                    "F1": ['FC1', 'F3', 'Fz'],
                                    "Fz": ['F1', 'F2'],
                                    "F2": ['FC2', 'Fz', 'F4'],
                                    "F4": ['FC4', 'F2'],
                                    "F6": [],
                                    "F8": [],
                                    "FT7": [],
                                    "FC5": [],
                                    "FC3": ['F3', 'C3', 'FC1'],
                                    "FC1": ['F1', 'C1', 'FC3'],
                                    "FC2": ['F2', 'C2', 'FC4'],
                                    "FC4": ['F4', 'C4', 'FC2'],
                                    "FC6": [],
                                    "FT8": [],
                                    "T7": [],
                                    "C5": [],
                                    "C3": ['FC3', 'CP3', 'C1'],
                                    "C1": ['FC1', 'CP1', 'C3', 'Cz'],
                                    "Cz": ['CPz', 'C1', 'C2'],
                                    "C2": ['FC2', 'CP2', 'Cz', 'C4'],
                                    "C4": ['FC4', 'CP4', 'C2'],
                                    "C6": [],
                                    "T8": [],
                                    "TP7": [],
                                    "CP5": [],
                                    "CP3": ['C3', 'CP1'],
                                    "CP1": ['C1', 'CP3', 'CPz'],
                                    "CPz": ['Cz', 'CP1', 'CP2'],
                                    "CP2": ['C2', 'CPz', 'CP4'],
                                    "CP4": ['C4', 'CP2'],
                                    "CP6": [],
                                    "TP8": [],
                                    "P7": [],
                                    "P5": [],
                                    "P3": [],
                                    "P1": [],
                                    "Pz": [],
                                    "P2": [],
                                    "P4": [],
                                    "P6": [],
                                    "P8": [],
                                    "PO7": [],
                                    "PO3": [],
                                    "POz": [],
                                    "PO4": [],
                                    "PO8": [],
                                    "O1": [],
                                    "Oz": [],
                                    "O2": []}
    if True:
        channel_neighbors = {
            # order: up-down-left-right
            "Fp1": ["AF7", "AF3", "AFz"],
            "Fp2": ["AF4", "AF8", "AFz"],
            "AF7": ["Fp1", "AF3", "F7"],
            "AF3": ["Fp1", "F3", "AF7", "AFz"],
            "AFz": ["Fp1", 'Fp2', "Fz", "AF3", "AF4"],
            "AF4": ["Fp2", "F4", "AFz", "AF8"],
            "AF8": ["Fp2", "AF4", "F8"],
            "F7": ["AF7", "FT7", "F5"],
            "F5": ["AF7", "AF3", "FC5", "F7", "F3"],
            "F3": ["AF3", "FC3", "F5", "F1"],
            "F1": ["AF3", "AFz", "FC1", "F3", "Fz"],
            "Fz": ["AFz", "F1", "F2"],
            "F2": ["AFz", "AF4", "FC2", "Fz", "F4"],
            "F4": ["AF4", "FC4", "F2", "F6"],
            "F6": ["AF4", "AF8", "FC6", "F4", "F8"],
            "F8": ["AF8", "FT8", "F6"],
            "FT7": ["F7", "T7", "FC5"],
            "FC5": ["F5", "C5", "FT7", "FC3"],
            "FC3": ["F3", "C3", "FC5", "FC1"],
            "FC1": ["F1", "C1", "FC3"],
            "FC2": ["F2", "C2", "FC4"],
            "FC4": ["F4", "C4", "FC2", "FC6"],
            "FC6": ["F6", "C6", "FC4", "FT8"],
            "FT8": ["F8", "T8", "FC6"],
            "T7": ["FT7", "TP7", "C5"],
            "C5": ["FC5", "CP5", "T7", "C3"],
            "C3": ["FC3", "CP3", "C5", "C1"],
            "C1": ["FC1", "CP1", "C3", "Cz"],
            "Cz": ["CPz", "C1", "C2"],
            "C2": ["FC2", "CP2", "Cz", "C4"],
            "C4": ["FC4", "CP4", "C2", "CP6"],
            "C6": ["FC6", "CP6", "C4", "T8"],
            "T8": ["FT8", "TP8", "C6"],
            "TP7": ["T7", "P7", "CP5"],
            "CP5": ["C5", "P5", "TP7", "CP3"],
            "CP3": ["C3", "P3", "CP5", "CP1"],
            "CP1": ["C1", "P1", "CP3", "CPz"],
            "CPz": ["Cz", "Pz", "CP1", "CP2"],
            "CP2": ["C2", "P2", "CPz", "CP4"],
            "CP4": ["C4", "P4", "CP2", "CP6"],
            "CP6": ["C6", "P6", "CP6", "TP8"],
            "TP8": ["T8", "P8", "CP6"],
            "P7": ["TP7", "PO7", "P5"],
            "P5": ["CP5", "PO7", "PO3", "P7", "P3"],
            "P3": ["CP3", "PO3", "P5", "P1"],
            "P1": ["CP1", "PO3", "POz", "P3", "Pz"],
            "Pz": ["CPz", "POz", "P1", "P2"],
            "P2": ["CP2", "POz", "PO4", "Pz", "P4"],
            "P4": ["CP4", "PO4", "P2", "P6"],
            "P6": ["CP6", "PO4", "PO8", "P4", "P8"],
            "P8": ["TP8", "PO8", "P6"],
            "PO7": ["P7", "O1", "PO3"],
            "PO3": ["P3", "O1", "Oz", "PO7", "POz"],
            "POz": ["Pz", "Oz", "PO3", "PO4"],
            "PO4": ["P4", "Oz", "O2", "POz", "PO8"],
            "PO8": ["P8", "O2", "PO4"],
            "O1": ["PO7", "PO3", "Oz"],
            "Oz": ["POz", "O1", "O2"],
            "O2": ["PO8", "PO4", "Oz"],
        }


    num_channels = len(ch_names)
    adj_matrix = np.zeros((num_channels, num_channels))

    for channel, neighbors in channel_neighbors.items():
        for neighbor in neighbors:
            i = ch_names.index(channel)
            j = ch_names.index(neighbor)
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1 
    
    np.fill_diagonal(adj_matrix, 1)

    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float).to(device)

    config = load_config('./config.yaml')
    config.eegnet.adj_matrix = adj_matrix_tensor
    config.eegnet.F1 = 8
    config.eegnet.D = 2
    config.eegnet.F2 = 16
    print('Using device: ', device)



    group = ''

    base = 'F:\Program Files\PycharmProjects\GPP_/'
    data_base = 'Z:\Datasets\GPP\data_brief/'

    sessions = os.listdir(data_base)
    print(data_base)
    band = '_0.1_48'
    # band = '_2_48'
    # for peri_remove in [1, 0, -1, 2]:
    for loss_name in ['mse', 'freq_reward_mse', 'freq_mse', 'reward_mse']:
        if peri_remove == 1:

            channel_select = 'InnerOnly{}'.format(group)
        elif peri_remove == -1:


            channel_select = 'PeriOnly{}'.format(group)
        elif peri_remove == 0:
            channel_select = 'All{}'.format(group)
        elif peri_remove == 2:
            channel_select = 'MotorOnly{}'.format(group)
        model_name = 'EEG2Gait' + '-' + loss_name

        if not os.path.exists(base + '/Results/{}/'.format(model_name)):
            os.mkdir(base + '/Results/{}/'.format(model_name))
        result_base = base + '/Results/{}/{}{}/'.format(model_name, channel_select, band)
        # result_base = r'F:\Program Files\PycharmProjects\GPP_\checkpoints\GaitNet\new24' + '/'
        existing = []
        if not os.path.exists(base + '/Results/{}/{}{}/'.format(model_name, channel_select, band)):
            os.mkdir(base + '/Results/{}/{}{}/'.format(model_name, channel_select, band))
        for file in os.listdir(result_base):
            if file[-3:] != '.pt':
                continue
            tmp_sbj = file.split('_')[1].strip()
            tmp_session = file.split('_')[2]
            TS = file.split('TS')[1][:3]
            if [tmp_sbj, tmp_session, TS] not in existing:
                existing.append([tmp_sbj, tmp_session, TS])
            else:
                a = 0
        existing = []
        print(len(existing))

        for sess in sessions:
            if len(sess.split('_')) < 2:
                continue
            sbj_name = sess.split('_')[0]

            session = sess.split('_')[1]


            mat_path = data_base + sess + '/walking_block_normal{}/'.format(band)
            used_joints = [i for i in range(6)]
            for step in range(200, 201, 20):
                if sbj_name != '12':
                    continue
                if [sbj_name, session, str(step)] in existing:
                    continue


                config.eegnet.eeg.time_step = step
                config.eegnet.num_chan_kin = 6
                config.eegnet.blk5_kernel = (step - config.eegnet.blk1_kernel) // 4
                run(step)
