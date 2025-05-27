import os
import numpy as np
import torch
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from gppdataset import GPPDataSet
from utils.utils import params_count
from utils.ymlfun import get_args, get_parser
from models.EEG2GAIT_model import EEG2GAIT
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from utils.myaml import load_config
from tqdm import tqdm
import time
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
    # if type(ind_act) is np.ndarray:
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
    
    batch_size = 100
    for i in tqdm(range(int(len(testdata) / batch_size))):
        s = i * batch_size
        e = i * batch_size + batch_size

        inputs = torch.tensor(testdata[s:e]).to(device)

        inputs = Variable(inputs)


        outputs = model(inputs).cpu().detach().numpy()
        if (i == 0):
            testPreJoints = outputs
        else:
            testPreJoints = np.vstack([testPreJoints, outputs])
            
    if (e != len(testdata)):
        inputs = torch.tensor(testdata[e:]).to(device)
        inputs = Variable(inputs)
        outputs = model(inputs).cpu().detach().numpy()
        testPreJoints = np.vstack([testPreJoints, outputs])
    rval = calc_rval(testTarget[:, [i for i in used_joints]], testPreJoints[:, [i for i in used_joints]])
    

    r2score = r2_score(testTarget[:, [i for i in used_joints]], testPreJoints[:, [i for i in used_joints]])
    if (domain != None):
        
        num_points = min(6000, testTarget.shape[0])
        t = np.arange(num_points)  
        
        plt.figure(figsize=(15, 10))  
        

        for i in range(6):
            plt.subplot(3, 2, i + 1)  
            y1 = testPreJoints[:num_points, i]
            y2 = testTarget[:num_points, i]

            plt.plot(t, y1, label='Predicted - Dimension {}'.format(i + 1))
            plt.plot(t, y2, label='Actual - Dimension {}'.format(i + 1))
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Comparison of Dimension {}'.format(i + 1))
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    return rval, r2score


def run(time_step):

    if not os.path.exists('./checkpoints/'.format(model_name)):
        os.mkdir('./checkpoints/'.format(model_name))
        
    count_flag = 0
    
    for p_id in range(0,1):
        
        

        for trial in range(2, 3):
            
            model = EEG2GAIT(config).to(device)
                

            if (count_flag == 0):
                num_param = params_count(model)
                count_flag = 1
            print('number of param of model:', num_param)
            batch_size = 100
            criterion = MSELoss()
            optimizer = optim.Adam(model.parameters())
            print(mat_path)
            dataset = GPPDataSet(mat_path, time_step, val_trial=config.eegnet.val_trial, test_trial=config.eegnet.test_trial)

            s_dic = dataset.get_data(laplacian=1, remove_EMG=peri_remove)
            trainEEG = s_dic['trainEEG']
            valEEG = s_dic['trainEEG']
            testEEG = s_dic['trainEEG']
            trainTarget = s_dic['trainTarget']
            valTarget = s_dic['trainTarget']
            testTarget = s_dic['trainTarget']
            scJoints = s_dic['scJoints']
            trainEEG = np.transpose(trainEEG, (0, 2, 1))  
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
            for epoch in range(config.num_teacher_epoch): 
                flag += 1
                if not os.path.exists('./checkpoints/' + model_name):
                    os.mkdir('./checkpoints/' + model_name)
                print("\nEpoch ", epoch)

                running_loss = 0.0
                print('start training', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                for i in tqdm(range(int(len(X_train) / batch_size))):
                    s = i * batch_size
                    e = i * batch_size + batch_size

                    inputs = torch.from_numpy(X_train[s:e]).to(device) 

                    labels = torch.FloatTensor(np.array([y_train[s:e]]) * 1.0).to(device)
                    labels = torch.squeeze(labels)
                    
                    inputs, labels = Variable(inputs), Variable(labels)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    optimizer.step()
                    

                    running_loss += loss.data

                # Validation accuracy
                print('finish training', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

                train_rval, train_r2score = evaluate(model, X_train, y_train[:, -6:])
                val_rval, val_r2score = evaluate(model, X_val, y_val[:, -6:])
                test_rval, test_r2score = evaluate(model, X_test, y_test[:, -6:])

                if (val_rval > highest_val):
                    save_path = result_base + '/{}_{}_{}_{}{}_TS{}_{}_'.format(channel_select, sbj_name, session, config.eegnet.val_trial, config.eegnet.test_trial, str(config.eegnet.eeg.time_step), band)

                    save_path = save_path + str(float(test_rval) * 100)[:5] + '.pt'
                    print(save_path)
                    best_model_state = model.state_dict()
                    flag = 0
                    print('to save')
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
                elif (epoch > 30):
                    print('to best')

                    print(best_dict['val_rval'], best_dict['test_rval'])
                print("Training Loss ", running_loss)
                print("Train PCC = {}, Train r2_score = {}".format(train_rval, train_r2score))
                print("val_ PCC = {}, val r2_score = {}".format(val_rval, val_r2score))
                print("Test PCC = {}, Test_r2_score = {}".format(test_rval, test_r2score))

        torch.save(best_dict, save_path)

        # start testing

def check(name, session, path):
    if (name in path and session in path):
        return True
    return False



if __name__ == '__main__':
    

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config = load_config('./config.yaml')
    config.eegnet.F1 = 8
    config.eegnet.D = 2
    config.eegnet.F2 = 16
    print('Using device: ', device)

    model_name = 'GaitNet'

    base = '../Datasets/GPP/data/'

    sessions = os.listdir(base)
    
    peri_remove = 0
    
    for band in ['_0.1_48']:
        if peri_remove == 1:

            channel_select = 'InnerOnly'
        elif peri_remove == -1:


            channel_select = 'PeriOnly'
        elif peri_remove == 0:
            channel_select = 'All'
        elif peri_remove == 2:
            channel_select = 'MotorOnly'
        result_base = '../Results/TapSize_analysis/{}/{}{}/'.format(model_name, channel_select, band)
        existing = []
        if not os.path.exists('../Results/TapSize_analysis/{}/{}{}/'.format(model_name, channel_select, band)):
            os.mkdir('../Results/TapSize_analysis/{}/{}{}/'.format(model_name, channel_select, band))
        for file in os.listdir(result_base):
            if file[-3:] != '.pt':
                continue
            tmp_sbj = file.split('_')[1].strip()
            tmp_session = file.split('_')[2]
            TS = file.split('TS')[1][:3]
            if [tmp_sbj, tmp_session, TS] not in existing:
                existing.append([tmp_sbj, tmp_session, TS])
            else:
                print([tmp_sbj, tmp_session, TS])
        existing = []
        for sess in sessions:
            if sess[0] == '0':
                continue

            if len(sess.split('_')) < 2:
                continue

            sbj_name = sess.split('_')[0]

            session = sess.split('_')[1]
            print(sbj_name, session)
            print(sess)


            mat_path = base + sess + '/eeg/walking_block_normal{}/'.format(band)



            used_joints = [i for i in range(6)]
            for step in range(100, 221, 20):
                if [sbj_name, session, str(step)] in existing:
                    print('existing', sbj_name, session)
                    continue

                config.eegnet.eeg.time_step = step
                config.eegnet.num_chan_kin = 6
                config.eegnet.blk5_kernel = (step - config.eegnet.blk1_kernel) // 4
                run(step)
