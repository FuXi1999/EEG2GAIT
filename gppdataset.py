from sklearn.utils import validation
import torch.utils.data as data
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time
from scipy.signal import filtfilt, butter
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import scipy
import mne

from utils.bandpass_filter import bandpassfilter_cheby2_sos


def int2twochar(s):
    s = '0' + str(s)
    s = s[-2:]
    return s

def prerpocess_eeg(eeg_data, lowcut=0.1, highcut=49.9, fs=1000, order=5):
    """

    Args:
        eeg_data: ndarray, (T,63), EEG signal
        lowcut:
        highcut:
        fs:
        order:

    Returns:
        filtered_and_downsampled_eeg_data: ndarray, (T, 63)

    """
    
    ch_names = ['ch'+str(i+1) for i in range(63)]
    ch_types = ['eeg'] * 63 
    
    
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)


    raw_data = mne.io.RawArray(data=eeg_data.T, info=info)
    
    raw_data.filter(lowcut, highcut, fir_design='firwin')
    tmp = raw_data.get_data()
    print(tmp.shape)
    
    decimation_factor = 10  
    
    raw_data.resample(sfreq=fs // decimation_factor)


    filtered_and_downsampled_eeg_data = np.transpose(raw_data.get_data())


    print(filtered_and_downsampled_eeg_data.shape)

    return filtered_and_downsampled_eeg_data

def downsample_signal(signal, fs=1000, decimation_factor=10):
    T = signal.shape[0]


    target_fs = fs / decimation_factor


    target_length = int(T * target_fs / fs)
    downsampled_signal = scipy.signal.resample(signal, target_length, axis=0)
    return downsampled_signal

class GPPDataSet():
    def __init__(self, mat_path, time_step, val_trial=1, test_trial=10, velocity=0, v_window=5, delay=0, model_name = 'GaitNet', bandCutF = None):
        """
        timestep: window_width
        separate_label: 0=extractor 1, 1=extractor 2
        """
        self.data_path = mat_path
        self.time_step = time_step

        self.velocity = velocity
        self.trial_len = []
        self.val_trial = val_trial
        self.test_trial = test_trial

        self.depay = delay


        self.ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                         'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                         'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
                         'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                         'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4',
                         'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.periphery_chans = ['AFz', 'Fp1', 'Fp2', 'AF7', 'AF8', 'F7', 'F8', 'FT7', 'FT8', 'T7',
                                'T8', 'TP7', 'TP8', 'P7', 'P8', 'PO7', 'PO8', 'O1', 'O2', 'Oz']
        self.motor_chans = ['Fz', 'F3', 'FC1', 'C3', 'CP1', 'CP2', 'Cz', 'C4', 'FC2', 'F4',
                            'F1', 'FC3', 'C1', 'CP3', 'CPz', 'CP4', 'C2', 'FC4', 'F2']
        self.peri_removed = False
        self.inner_removed = False
        self.motor_only = False

        self.model_name = model_name
        self.bandCutF = bandCutF
        self.channel_neighbors = {
            #order: up-down-left-right
            "Fp1": ["AF7", "AF3", "AFz"],
            "Fp2": ["AF4", "AF8", "AFz"],

            "AF7": ["Fp1", "AF3", "F7"],
            "AF3": ["Fp1", "F3", "AF7", "AFz"],
            "AFz": ["Fp1",'Fp2', "Fz", "AF3", "AF4"],
            "AF4": ["Fp2", "F4", "AFz", "AF8"],
            "AF8": ["Fp2", "AF4", "F8"],

            "F7": ["AF7", "FT7",        "F5"],
            "F5": ["AF7", "AF3", "FC5", "F7", "F3"],
            "F3": ["AF3", "FC3", "F5", "F1"],
            "F1": ["AF3", "AFz", "FC1", "F3", "Fz"],
            "Fz": ["AFz",         "F1", "F2"],
            "F2": ["AFz", "AF4", "FC2", "Fz", "F4"],
            "F4": ["AF4", "FC4", "F2", "F6"],
            "F6": ["AF4", "AF8", "FC6", "F4", "F8"],
            "F8": ["AF8", "FT8", "F6"],

            "FT7": ["F7", "T7",       "FC5"],
            "FC5": ["F5", "C5", "FT7", "FC3"],
            "FC3": ["F3", "C3", "FC5", "FC1"],
            "FC1": ["F1", "C1", "FC3"],
            "FC2": ["F2", "C2", "FC4"],
            "FC4": ["F4", "C4", "FC2", "FC6"],
            "FC6": ["F6", "C6", "FC4", "FT8"],
            "FT8": ["F8", "T8", "FC6"],

            "T7": ["FT7", "TP7",       "C5"],
            "C5": ["FC5", "CP5", "T7", "C3"],
            "C3": ["FC3", "CP3", "C5", "C1"],
            "C1": ["FC1", "CP1", "C3", "Cz"],
            "Cz": ["CPz", "C1", "C2"],
            "C2": ["FC2", "CP2", "Cz", "C4"],
            "C4": ["FC4", "CP4", "C2", "CP6"],
            "C6": ["FC6", "CP6", "C4", "T8"],
            "T8": ["FT8", "TP8", "C6"],

            "TP7": ["T7", "P7",       "CP5"],
            "CP5": ["C5", "P5", "TP7", "CP3"],
            "CP3": ["C3", "P3", "CP5", "CP1"],
            "CP1": ["C1", "P1", "CP3", "CPz"],
            "CPz": ["Cz", "Pz", "CP1", "CP2"],
            "CP2": ["C2", "P2", "CPz", "CP4"],
            "CP4": ["C4", "P4", "CP2", "CP6"],
            "CP6": ["C6", "P6", "CP6", "TP8"],
            "TP8": ["T8", "P8", "CP6"],

            "P7": ["TP7", "PO7",      "P5"],
            "P5": ["CP5", "PO7", "PO3", "P7", "P3"],
            "P3": ["CP3", "PO3", "P5", "P1"],
            "P1": ["CP1", "PO3", "POz", "P3", "Pz"],
            "Pz": ["CPz", "POz", "P1", "P2"],
            "P2": ["CP2", "POz", "PO4", "Pz", "P4"],
            "P4": ["CP4", "PO4", "P2", "P6"],
            "P6": ["CP6", "PO4", "PO8", "P4", "P8"],
            "P8": ["TP8", "PO8", "P6"],

            "PO7": ["P7", "O1",        "PO3"],
            "PO3": ["P3", "O1", "Oz", "PO7", "POz"],
            "POz": ["Pz", "Oz", "PO3", "PO4"],
            "PO4": ["P4", "Oz", "O2", "POz", "PO8"],
            "PO8": ["P8", "O2", "PO4"],

            "O1": ["PO7", "PO3", "Oz"],
            "Oz": ["POz", "O1", "O2"],
            "O2": ["PO8", "PO4", "Oz"],

        }
        self.peri_neighbors = {
            "Fp1": ['AF7', 'AFz'],
            "Fp2": ['AF8', 'AFz'],
            "AF7": ['Fp1', 'F7'],
            "AF3": [],
            "AFz": ['Fp1', 'Fp2'],
            "AF4": [],
            "AF8": ['Fp2', 'F8'],
            "F7": ['AF7', 'FT7'],
            "F5": [],
            "F3": [],
            "F1": [],
            "Fz": [],
            "F2": [],
            "F4": [],
            "F6": [],
            "F8": ['AF8', 'FT8'],
            "FT7": ['F7', 'T7'],
            "FC5": [],
            "FC3": [],
            "FC1": [],
            "FC2": [],
            "FC4": [],
            "FC6": [],
            "FT8": ['F8', 'T8'],
            "T7": ['FT7', 'TP7'],
            "C5": [],
            "C3": [],
            "C1": [],
            "Cz": [],
            "C2": [],
            "C4": [],
            "C6": [],
            "T8": ['FT8', 'TP8'],
            "TP7": ['T7', 'P7'],
            "CP5": [],
            "CP3": [],
            "CP1": [],
            "CPz": [],
            "CP2": [],
            "CP4": [],
            "CP6": [],
            "TP8": ['T8', 'P8'],
            "P7": ['TP7', 'PO7'],
            "P5": [],
            "P3": [],
            "P1": [],
            "Pz": [],
            "P2": [],
            "P4": [],
            "P6": [],
            "P8": ['TP8', 'PO8'],
            "PO7": ['P7', 'O1'],
            "PO3": [],
            "POz": [],
            "PO4": [],
            "PO8": ['P8', 'O2'],
            "O1": ['PO7', 'Oz'],
            "Oz": ['O1', 'O2'],
            "O2": ['PO8', 'Oz']
        }
        self.motor_neighbors = {"Fp1": [],
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
        self.inner_neighbors = {"Fp1":  [],
                                "Fp2":  [],
                                "AF7":  [],
                                "AF3":  ['F3', 'AFz'],
                                "AFz":  [],
                                "AF4":  ['F4', 'AF8'],
                                "AF8":  [],
                                "F7":  [],
                                "F5":  ['AF3', 'FC5', 'F3'],
                                "F3":  ['AF3', 'FC3', 'F5', 'F1'],
                                "F1":  ['AF3', 'FC1', 'F3', 'Fz'],
                                "Fz":  ['F1', 'F2'],
                                "F2":  ['AF4', 'FC2', 'Fz', 'F4'],
                                "F4":  ['AF4', 'FC4', 'F2', 'F6'],
                                "F6":  ['AF4', 'FC6', 'F4'],
                                "F8":  [],
                                "FT7":  [],
                                "FC5":  ['F5', 'C5', 'FC3'],
                                "FC3":  ['F3', 'C3', 'FC5', 'FC1'],
                                "FC1":  ['F1', 'C1', 'FC3'],
                                "FC2":  ['F2', 'C2', 'FC4'],
                                "FC4":  ['F4', 'C4', 'FC2', 'FC6'],
                                "FC6":  ['F6', 'C6', 'FC4'],
                                "FT8":  [],
                                "T7":  [],
                                "C5":  ['FC5', 'CP5', 'C3'],
                                "C3":  ['FC3', 'CP3', 'C5', 'C1'],
                                "C1":  ['FC1', 'CP1', 'C3', 'Cz'],
                                "Cz":  ['CPz', 'C1', 'C2'],
                                "C2":  ['FC2', 'CP2', 'Cz', 'C4'],
                                "C4":  ['FC4', 'CP4', 'C2', 'CP6'],
                                "C6":  ['FC6', 'CP6', 'C4'],
                                "T8":  [],
                                "TP7":  [],
                                "CP5":  ['C5', 'P5', 'CP3'],
                                "CP3":  ['C3', 'P3', 'CP5', 'CP1'],
                                "CP1":  ['C1', 'P1', 'CP3', 'CPz'],
                                "CPz":  ['Cz', 'Pz', 'CP1', 'CP2'],
                                "CP2":  ['C2', 'P2', 'CPz', 'CP4'],
                                "CP4":  ['C4', 'P4', 'CP2', 'CP6'],
                                "CP6":  ['C6', 'P6', 'CP6'],
                                "TP8":  [],
                                "P7":  [],
                                "P5":  ['CP5', 'PO3', 'P3'],
                                "P3":  ['CP3', 'PO3', 'P5', 'P1'],
                                "P1":  ['CP1', 'PO3', 'POz', 'P3', 'Pz'],
                                "Pz":  ['CPz', 'POz', 'P1', 'P2'],
                                "P2":  ['CP2', 'POz', 'PO4', 'Pz', 'P4'],
                                "P4":  ['CP4', 'PO4', 'P2', 'P6'],
                                "P6":  ['CP6', 'PO4', 'P4'],
                                "P8":  [],
                                "PO7":  [],
                                "PO3":  ['P3', 'Oz', 'POz'],
                                "POz":  ['Pz', 'PO3', 'PO4'],
                                "PO4":  ['P4', 'O2', 'POz'],
                                "PO8":  [],
                                "O1":  [],
                                "Oz":  [],
                                "O2":  []}
        self.train_flag = [0]
        self.val_flag = [0]
        self.test_flag = [0]
        self.se_cut = 1



    def get_data(self, remove_EMG=0, laplacian=0, get_seq = 0, preprocessed=1, get_trialwise = 0):


        self.eegdata = np.empty((0, 59))
        self.kindata = np.empty((0, 6))
        mult = 1
        val_trial = self.val_trial
        test_trial = self.test_trial
        mat_list = os.listdir(self.data_path)
        mat_list.sort(key=lambda x: int(x[5:-4]))
        
        for matfile in mat_list:
            print(matfile)
            mat = sio.loadmat(self.data_path + matfile)
            mat['eeg_data'] = mat['eeg_data'][self.se_cut:-self.se_cut,:]
            mat['goniometer_data'] = mat['goniometer_data'][self.se_cut:-self.se_cut,:]
            if (preprocessed == 1):
                self.trial_len.append(mat['eeg_data'].shape[0])
            if mat['goniometer_data'].shape[1] == 12:
                mat['goniometer_data'] = mat['goniometer_data'][:,[0,2,4,6,8,10]]
                
            if mat['goniometer_data'].shape[1] == 8:
                mat['goniometer_data'] = mat['goniometer_data'][:,:6]
            self.eegdata = np.concatenate((self.eegdata, mat['eeg_data']), axis=0).astype(np.float32)
            self.kindata = np.concatenate((self.kindata, mat['goniometer_data']), axis=0).astype(np.float32)

        if (preprocessed == 0):
            self.eegdata = prerpocess_eeg(self.eegdata, lowcut=0.1, highcut=49.9, fs=1000)
            
            self.kindata = downsample_signal(self.kindata, decimation_factor=10)

        ttmp = self.kindata[:,0]
        sorted_ttmp = np.sort(ttmp, axis=0)

        # Calculate cumulative probabilities
        cumulative_probs = np.arange(1, len(sorted_ttmp) + 1) / len(sorted_ttmp)
        idx_0_01 = np.argmax(cumulative_probs > 0.01) - 1
        idx_0_99 = np.argmax(cumulative_probs > 0.99) - 1

        # These are the indices where the conditions are satisfied
        print(sorted_ttmp[idx_0_01], sorted_ttmp[idx_0_99])


        train_len = sum(self.trial_len[:-(val_trial + test_trial)])
        val_len = sum(self.trial_len[-(val_trial + test_trial) : -test_trial])
        test_len = sum(self.trial_len[-test_trial:])
        trainEEG = self.eegdata[:train_len * mult, :].astype(np.float32)
        trainJoints = self.kindata[:train_len, :].astype(np.float32)
        valEEG = self.eegdata[train_len * mult:(train_len + val_len) * mult, :].astype(np.float32)
        valJoints = self.kindata[train_len:train_len + val_len, :].astype(np.float32)
        testEEG = self.eegdata[(train_len + val_len) * mult:, :].astype(np.float32)
        testJoints = self.kindata[train_len + val_len:, :].astype(np.float32)
        print('standardize_dataset', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        _, trainEEG, \
            valEEG, testEEG = self.standardize_dataset(trainEEG, valEEG, testEEG) #(T, C)standarize along channel
        scJoints, trainJoints, valJoints, \
            testJoints = self.standardize_dataset(trainJoints, valJoints, testJoints)# (T,J)
        print('finish standardize_dataset', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        self.eegdata = np.concatenate((trainEEG, valEEG, testEEG), axis=0)
        self.kindata = np.concatenate((trainJoints, valJoints, testJoints), axis=0)
        
        if (get_seq == 1):
            return_dic = {'trainEEG': trainEEG, 'trainTarget': trainJoints,
                          'valEEG': valEEG,
                          'valTarget': valJoints, 'testEEG': testEEG,
                          'testTarget': testJoints,
                          'scJoints': scJoints}
            return return_dic

        flag = 0
        if get_trialwise == 1:
            trainEEG = []
            valEEG = []
            testEEG = []
            trainTarget = []
            valTarget = []
            testTarget = []
            for i in range(len(self.trial_len)):
                l = self.trial_len[i]
                tmpeeg = self.eegdata[flag * mult:(flag + l) * mult]
                tmpjoints = self.kindata[flag:flag + l, :]
                if (remove_EMG == 1):
                    tmpeeg = self.remove_periphery_EMG(tmpeeg)
                    self.peri_removed = True
                elif (remove_EMG == -1):
                    tmpeeg = self.remove_inner_EMG(tmpeeg)
                    self.inner_removed = True
                elif (remove_EMG == 2):
                    tmpeeg = self.keep_motor_only(tmpeeg)
                    self.motor_only = True
                if (laplacian):
                    tmpeeg = self.laplacian_filter(tmpeeg)
                EEG_samp, Target_samp = \
                    self.make_dataset(tmpeeg, tmpjoints, self.time_step)
                if (i < len(self.trial_len)):
                    
                    trainEEG.append(EEG_samp)
                    trainTarget.append(Target_samp)
                    
                flag += l
            print('finish test making', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            return_dic = {'trainEEG': trainEEG, 'trainTarget': trainTarget,
                          'scJoints': scJoints}
            return return_dic

        trainTarget_make = []
        trainEEG_make = []
        print('start making dataset', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for i in range(len(self.trial_len)):
            l = self.trial_len[i]
            tmpeeg = self.eegdata[flag * mult:(flag + l) * mult]
            if (remove_EMG == 1):
                tmpeeg = self.remove_periphery_EMG(tmpeeg)
                self.peri_removed = True
            elif (remove_EMG == -1):
                tmpeeg = self.remove_inner_EMG(tmpeeg)
                self.inner_removed = True
            tmpjoints = self.kindata[flag:flag + l, :]
            if (laplacian):
                tmpeeg = self.laplacian_filter(tmpeeg)


            EEG_samp, Target_samp = \
                self.make_dataset(tmpeeg, tmpjoints, self.time_step)
            
            if (i < len(self.trial_len) - val_trial - test_trial):
                
                if (i == 0):
                    trainEEG_make = EEG_samp
                    trainTarget_make = Target_samp
                else:
                    trainEEG_make = np.concatenate((trainEEG_make, EEG_samp), axis=0)
                    trainTarget_make = np.concatenate((trainTarget_make, Target_samp), axis=0)
                self.train_flag.append(trainTarget_make.shape[0])
            elif (len(self.trial_len) - val_trial - test_trial <= i < len(self.trial_len) - test_trial):
                
                if (i == len(self.trial_len) - val_trial - test_trial):
                    valEEG_make = EEG_samp
                    valTarget_make = Target_samp
                else:
                    valEEG_make = np.concatenate((valEEG_make, EEG_samp), axis=0)
                    valTarget_make = np.concatenate((valTarget_make, Target_samp), axis=0)
                self.val_flag.append(valTarget_make.shape[0])
            elif (i >= len(self.trial_len) - test_trial):
                
                if (i == len(self.trial_len) - test_trial):
                    testEEG_make = EEG_samp
                    testTarget_make = Target_samp
                else:
                    testEEG_make = np.concatenate((testEEG_make, EEG_samp), axis=0)
                    testTarget_make = np.concatenate((testTarget_make, Target_samp), axis=0)

                self.test_flag.append(testTarget_make.shape[0])
            flag += l
        print('finish test making', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        return_dic = {'trainEEG': trainEEG_make, 'trainTarget': trainTarget_make,
                      'valEEG': valEEG_make,
                      'valTarget': valTarget_make, 'testEEG': testEEG_make,
                      'testTarget': testTarget_make,
                      'train_flag': self.train_flag,
                      'val_flag': self.val_flag,
                      'test_flag': self.test_flag,
                      'scJoints': scJoints}
        
        return return_dic

    def make_dataset(self, eeg, joints, time_step, domain = 0):
        '''
        create dataset
        :param eeg:(numpy, [n_samples, n_chans]) eeg matrix
        :param joints:(numpy, [n_samples, n_chans]) joints matrix
        :param time_step:(int) window size
        :parm domain:(str) flag of train/validation/test
        '''
        
        Jtimestep = time_step
        data_len = int(joints.shape[0] - Jtimestep + 1)
        eeg_chans_num = eeg.shape[1]
        

        fat_EEG = np.full((data_len, time_step, eeg_chans_num), np.nan).astype(np.float32)
        for idx in range(data_len):
            start = idx
            end = idx + time_step
            fat_EEG[idx] = eeg[start:end, :].astype(np.float32)
            

        joints_cor = joints[Jtimestep - 1:, :].astype(np.float32)

        return fat_EEG, joints_cor

    def standardize_dataset(self, train_data, validation_data, test_data):
        """
        Standardize the dataset.
        based on https://github.com/shonaka/EEG-neural-decoding
        :param sc: standardize class from sklearn "StandardScaler()"
        :param train_data: the train data you want to standardize
        :param validation_data: the validation data you want to standardize based on the train data
        :param test_data: the test data you want to standardize based on the train data
        Returns:
        :param sc: standardize class fit to the data. Used later for transformation.
        :param train_stan: standardized train data
        :param validation_stan:  standardized validation data
        :param test_stan:  standardized test data
        """
        sc = StandardScaler()
        train_stan = sc.fit_transform(train_data)
        validation_stan = sc.transform(validation_data)
        test_stan = sc.transform(test_data)

        return sc, train_stan, validation_stan, test_stan
    def remove_periphery_EMG(self, eeg):
        """

        :param eeg: T x num_of_Channels
        :return:
        """
        ch_names = self.ch_names


        periphery = []
        internal = []
        for i in range(len(ch_names)):
            if ch_names[i] in self.periphery_chans:
                periphery.append(i)
            else:
                internal.append(i)
        eeg_tmp1 = eeg[:, internal]
        eeg_tmp3 = np.zeros_like(eeg[:, periphery])
        # for j in range(len(periphery)):
        #     eeg_tmp3[:, j] = 4 * eeg_tmp2[:, j] - (eeg_tmp2[:, j - 1] + eeg_tmp2[:, (j + 1) % len(periphery_chans)] + eeg_tmp2[:, j - 2] + eeg_tmp2[:, (j + 2) % len(periphery_chans)])
        new_eeg = np.zeros_like(eeg)
        for i in range(len(ch_names)):
            if (i in periphery):
                new_eeg[:,i] = eeg_tmp3[:, periphery.index(i)]
            else:
                new_eeg[:, i] = eeg_tmp1[:, internal.index(i)]
        return new_eeg
    def keep_motor_only(self, eeg):
        """

        :param eeg: T x num_of_Channels
        :return:
        """
        ch_names = self.ch_names


        motor = []
        others = []
        for i in range(len(ch_names)):
            if ch_names[i] in self.motor_chans:
                motor.append(i)
            else:
                others.append(i)
        eeg_tmp1 = eeg[:, motor]
        eeg_tmp3 = np.zeros_like(eeg[:, others])
        # for j in range(len(periphery)):
        #     eeg_tmp3[:, j] = 4 * eeg_tmp2[:, j] - (eeg_tmp2[:, j - 1] + eeg_tmp2[:, (j + 1) % len(periphery_chans)] + eeg_tmp2[:, j - 2] + eeg_tmp2[:, (j + 2) % len(periphery_chans)])
        new_eeg = np.zeros_like(eeg)
        for i in range(len(ch_names)):
            if (i in others):
                new_eeg[:,i] = eeg_tmp3[:, others.index(i)]
            else:
                new_eeg[:, i] = eeg_tmp1[:, motor.index(i)]
        return new_eeg
    def remove_inner_EMG(self, eeg):
        """

        :param eeg: T x num_of_Channels
        :return:
        """
        ch_names = self.ch_names


        periphery = []
        internal = []
        for i in range(len(ch_names)):
            if ch_names[i] in self.periphery_chans:
                periphery.append(i)
            else:
                internal.append(i)
        eeg_tmp1 = eeg[:, periphery]
        eeg_tmp3 = np.zeros_like(eeg[:, internal])
        # for j in range(len(periphery)):
        #     eeg_tmp3[:, j] = 4 * eeg_tmp2[:, j] - (eeg_tmp2[:, j - 1] + eeg_tmp2[:, (j + 1) % len(periphery_chans)] + eeg_tmp2[:, j - 2] + eeg_tmp2[:, (j + 2) % len(periphery_chans)])
        new_eeg = np.zeros_like(eeg)
        for i in range(len(ch_names)):
            if (i in internal):
                new_eeg[:,i] = eeg_tmp3[:, internal.index(i)]
            else:
                new_eeg[:, i] = eeg_tmp1[:, periphery.index(i)]
        return new_eeg
    def laplacian_filter(self, eeg):
        eeg_dict = {}
        if self.peri_removed:
            # for chan in self.ch_names:
            #     if chan in self.periphery_chans:
            #         self.ch_names.remove(chan)
            self.channel_neighbors = self.inner_neighbors
        if self.inner_removed:
            self.channel_neighbors = self.peri_neighbors
        if self.motor_only:
            self.channel_neighbors = self.motor_neighbors
        for i in range (eeg.shape[1]):
            eeg_dict[self.ch_names[i]] = eeg[:,i]
        for chan in eeg_dict:
            eeg_dict[chan] = eeg_dict[chan] * len(self.channel_neighbors[chan])
            for i in range(len(self.channel_neighbors[chan])):
                eeg_dict[chan] = eeg_dict[chan] - eeg_dict[self.channel_neighbors[chan][i]]
        new_eeg = np.zeros_like(eeg)
        for i in range(len(self.ch_names)):
            new_eeg[:,i] = eeg_dict[self.ch_names[i]]
        return new_eeg
    def bandpass_filter(self, eeg, bandFiltCutF):
        eeg = np.transpose(eeg)
        eeg = np.expand_dims(eeg, 0)
        for j in range(len(bandFiltCutF)):
            band = bandFiltCutF[j]
            tmp = bandpassfilter_cheby2_sos(eeg, band)
            if (j == 0):
                x_tmp = np.expand_dims(tmp, 1)
            else:
                x_tmp = np.concatenate((x_tmp, np.expand_dims(tmp, 1)), axis=1)
        x = x_tmp.astype('float32')
        x = np.squeeze(x)
        x = np.transpose(x,(2,1,0))
        return x
