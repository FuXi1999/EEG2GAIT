sub_dir = '..\Datasets\GPP\data\session_name\';
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadbv();
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','raw','gui','off'); 
EEG=pop_chanedit(EEG, []);
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
eeglab redraw;
% need to load the eeg data using
event_stimuli_length = length(EEG.event);

%load the goniometer
filename = sub_dir + "\goniometer\goniometers.mat";
load(filename);

% find the trigger point in the goniometers
max_value = max(Goniometers(:,13));
Raw_trigger_idx = find(Goniometers(:,13)==max_value);

trigger_time_goniometers = [];
for idx = 1:length(Raw_trigger_idx)
    temp_timestamp = Raw_trigger_idx(idx);
    
    % get the first timestamps for each clicking
    if Goniometers(temp_timestamp-1,13)<0
        trigger_time_goniometers = [trigger_time_goniometers; temp_timestamp];
    end
end
% get the start time of walking session in the trigger
if trigger_time_goniometers(end)- trigger_time_goniometers(end-1)<10^6
    trigger_walking_session_start_time = trigger_time_goniometers(end-3)
else
    trigger_walking_session_start_time = trigger_time_goniometers(end-2)
end

trigger_time_eeg_s11 = [];
for e_i = 1:event_stimuli_length
    if strcmp(EEG.event(e_i).type, 'S 11')
        trigger_time_eeg_s11 = [trigger_time_eeg_s11; EEG.event(e_i).latency];    
    end
end
% get the start time of the walking session in EEG
if trigger_time_eeg_s11(end) > trigger_time_goniometers(end)
    walking_session_start_time = trigger_time_eeg_s11(end-2);
else
    walking_session_start_time = trigger_time_eeg_s11(end-1); 
end

% Calculate the time difference according to the trigger point of test
% walking
goniometer_time_off_set = trigger_walking_session_start_time-walking_session_start_time;

trigger_time_eeg_s10 = [];
for e_i = 1:event_stimuli_length
    if strcmp(EEG.event(e_i).type, 'S 10')
        trigger_time_eeg_s10 = [trigger_time_eeg_s10; EEG.event(e_i).latency];    
    end
end
% get the end time of walking and non-walking blocks
walking_block_end_time_list = [];
for idx = 1:length(trigger_time_eeg_s10)
    if trigger_time_eeg_s10(idx)>walking_session_start_time
        walking_block_end_time_list = [walking_block_end_time_list; trigger_time_eeg_s10(idx)];
    end
end

trigger_time_eeg_s1= [];
for e_i = 1:event_stimuli_length
    if strcmp(EEG.event(e_i).type, 'S  1')
        trigger_time_eeg_s1 = [trigger_time_eeg_s1; EEG.event(e_i).latency];    
    end
end
trigger_time_eeg_s2= [];
for e_i = 1:event_stimuli_length
    if strcmp(EEG.event(e_i).type, 'S  2')
        trigger_time_eeg_s2 = [trigger_time_eeg_s2; EEG.event(e_i).latency];    
    end
end
trigger_time_eeg_s3 = [];
for e_i = 1:event_stimuli_length
    if strcmp(EEG.event(e_i).type, 'S  3')
        trigger_time_eeg_s3 = [trigger_time_eeg_s3; EEG.event(e_i).latency];    
    end
end

% get the start time and end time for each walking block
walking_block_test_time = [walking_session_start_time, walking_block_end_time_list(1)];
walking_block1_time = [trigger_time_eeg_s1(1), walking_block_end_time_list(2)];
walking_abnormal_left_time = [trigger_time_eeg_s2(1), walking_block_end_time_list(3)];
walking_block2_time = [trigger_time_eeg_s1(2), walking_block_end_time_list(4)];
walking_abnormal_right_time = [trigger_time_eeg_s3(1), walking_block_end_time_list(5)];
walking_block3_time = [trigger_time_eeg_s1(3), walking_block_end_time_list(6)];


%find the trigger time for each trial
trial_time = [];
trial_temp_1 = [];
trial_temp_2 = [];
for e_i = 1:event_stimuli_length
    
    if strcmp(EEG.event(e_i).type, 'S 12')
        trial_temp_1 = [trial_temp_1, EEG.event(e_i).latency];
    
    elseif strcmp(EEG.event(e_i).type, 'S 13')
        trial_temp_1 = [trial_temp_1, EEG.event(e_i).latency];
        trial_time = [trial_time; trial_temp_1];
        trial_temp_1 = [];
    
    elseif strcmp(EEG.event(e_i).type, 'S 14')
        trial_temp_2 = [trial_temp_2, EEG.event(e_i).latency];
    
    elseif strcmp(EEG.event(e_i).type, 'S 15')
        trial_temp_2 = [trial_temp_2, EEG.event(e_i).latency];
        trial_time = [trial_time; trial_temp_2];
        trial_temp_2 = [];
    end
end





% plot the time point for double check
h = figure(2);
plot(Goniometers(:,end))
hold on;
temp_trigger_time_eeg_s1 = [trigger_time_eeg_s1, ones(size(trigger_time_eeg_s1))];
scatter(temp_trigger_time_eeg_s1(:,1),temp_trigger_time_eeg_s1(:,2));
temp_trigger_time_eeg_s2 = [trigger_time_eeg_s2, ones(size(trigger_time_eeg_s2))];
scatter(temp_trigger_time_eeg_s2(:,1),temp_trigger_time_eeg_s2(:,2),"*");
temp_trigger_time_eeg_s3 = [trigger_time_eeg_s3, ones(size(trigger_time_eeg_s3))];
scatter(temp_trigger_time_eeg_s3(:,1),temp_trigger_time_eeg_s3(:,2),"x");
temp_trigger_time_eeg_s10 = [trigger_time_eeg_s10, ones(size(trigger_time_eeg_s10))];
scatter(temp_trigger_time_eeg_s10(:,1),temp_trigger_time_eeg_s10(:,2),"g^");
temp_trigger_time_eeg_s11 = [trigger_time_eeg_s11, ones(size(trigger_time_eeg_s11))];
scatter(temp_trigger_time_eeg_s11(:,1),temp_trigger_time_eeg_s11(:,2),"b>");
filename = sub_dir + "Time_point.fig";
savefig(h, filename);

%% walking block 1
% get the entire block 
eeg_data_all = EEG.data(:,walking_block1_time(1):walking_block1_time(2));
goniometer_data_all = Goniometers(walking_block1_time(1)+goniometer_time_off_set:walking_block1_time(2)+goniometer_time_off_set,:);
% get the relative timestamps of trials in each session and partition both
% EEG data and goniometers
temp_trial_time_idx = find(walking_block1_time(1)<trial_time(:,1) & trial_time(:,1)<walking_block1_time(2));
current_trial_time_eeg = trial_time(temp_trial_time_idx,:);
current_trial_time = current_trial_time_eeg-walking_block1_time(1);
% save the data
filename = sub_dir + "\Partitioned_session_walking_block_1.mat";
save(filename,'eeg_data_all','goniometer_data_all','current_trial_time');
filename = sub_dir + "\_Partitioned_session_walking_block_1.mat";
save(filename,'eeg_data_all');

            
%% walking block 2
% get the entire block 
eeg_data_all = EEG.data(:,walking_block2_time(1):walking_block2_time(2));
goniometer_data_all = Goniometers(walking_block2_time(1)+goniometer_time_off_set:walking_block2_time(2)+goniometer_time_off_set,:);
% get the relative timestamps of trials in each session and partition both
% EEG data and goniometers
temp_trial_time_idx = find(walking_block2_time(1)<trial_time(:,1) & trial_time(:,1)<walking_block2_time(2));
current_trial_time_eeg = trial_time(temp_trial_time_idx,:);
current_trial_time = current_trial_time_eeg-walking_block2_time(1);
% save the data
filename = sub_dir + "\Partitioned_session_walking_block_2.mat";
save(filename,'eeg_data_all','goniometer_data_all','current_trial_time');
filename = sub_dir + "\_Partitioned_session_walking_block_2.mat";
save(filename,'eeg_data_all');

%% walking block 3
% get the entire block 
eeg_data_all = EEG.data(:,walking_block3_time(1):walking_block3_time(2));
goniometer_data_all = Goniometers(walking_block3_time(1)+goniometer_time_off_set:walking_block3_time(2)+goniometer_time_off_set,:);
% get the relative timestamps of trials in each session and partition both
% EEG data and goniometers
temp_trial_time_idx = find(walking_block3_time(1)<trial_time(:,1) & trial_time(:,1)<walking_block3_time(2));
current_trial_time_eeg = trial_time(temp_trial_time_idx,:);
current_trial_time = current_trial_time_eeg-walking_block3_time(1);
% save the data
filename = sub_dir + "\Partitioned_session_walking_block_3.mat";
save(filename,'eeg_data_all','goniometer_data_all','current_trial_time');
filename = sub_dir + "\_Partitioned_session_walking_block_3.mat";
save(filename,'eeg_data_all');

%% walking abnormal left
% get the entire block 
eeg_data_all = EEG.data(:,walking_abnormal_left_time(1):walking_abnormal_left_time(2));
goniometer_data_all = Goniometers(walking_abnormal_left_time(1)+goniometer_time_off_set:walking_abnormal_left_time(2)+goniometer_time_off_set,:);
% get the relative timestamps of trials in each session and partition both
% EEG data and goniometers
temp_trial_time_idx = find(walking_abnormal_left_time(1)<trial_time(:,1) & trial_time(:,1)<walking_abnormal_left_time(2));
current_trial_time_eeg = trial_time(temp_trial_time_idx,:);
current_trial_time = current_trial_time_eeg-walking_abnormal_left_time(1);
% save the data
filename = sub_dir + "\Partitioned_session_walking_block_left.mat";
save(filename,'eeg_data_all','goniometer_data_all','current_trial_time');
filename = sub_dir + "\_Partitioned_session_walking_block_left.mat";
save(filename,'eeg_data_all');
            
%% walking abnormal right
% get the entire block 
eeg_data_all = EEG.data(:,walking_abnormal_right_time(1):walking_abnormal_right_time(2));
goniometer_data_all = Goniometers(walking_abnormal_right_time(1)+goniometer_time_off_set:walking_abnormal_right_time(2)+goniometer_time_off_set,:);
% get the relative timestamps of trials in each session and partition both
% EEG data and goniometers
temp_trial_time_idx = find(walking_abnormal_right_time(1)<trial_time(:,1) & trial_time(:,1)<walking_abnormal_right_time(2));
current_trial_time_eeg = trial_time(temp_trial_time_idx,:);
current_trial_time = current_trial_time_eeg-walking_abnormal_right_time(1);
% save the data
filename = sub_dir + "\Partitioned_session_walking_block_right.mat";
save(filename,'eeg_data_all','goniometer_data_all','current_trial_time');
filename = sub_dir + "\_Partitioned_session_walking_block_right.mat";
save(filename,'eeg_data_all');
%% walking abnormal left with 2 sandboxes
% get the entire block 
eeg_data_all = EEG.data(:,walking_abnormal_left_time2(1):walking_abnormal_left_time2(2));
goniometer_data_all = Goniometers(walking_abnormal_left_time2(1)+goniometer_time_off_set:walking_abnormal_left_time2(2)+goniometer_time_off_set,:);
% get the relative timestamps of trials in each session and partition both
% EEG data and goniometers
temp_trial_time_idx = find(walking_abnormal_left_time2(1)<trial_time(:,1) & trial_time(:,1)<walking_abnormal_left_time2(2));
current_trial_time_eeg = trial_time(temp_trial_time_idx,:);
current_trial_time = current_trial_time_eeg-walking_abnormal_left_time2(1);
% save the data
filename = sub_dir + "Partitioned_session_walking_left_2sand.mat";
save(filename,'eeg_data_all','goniometer_data_all','current_trial_time');

%% walking test
% get the entire block 
eeg_data_all = EEG.data(:,walking_block_test_time(1):walking_block_test_time(2));
goniometer_data_all = Goniometers(walking_block_test_time(1)+goniometer_time_off_set:walking_block_test_time(2)+goniometer_time_off_set,:);
% get the relative timestamps of trials in each session and partition both
% EEG data and goniometers
temp_trial_time_idx = find(walking_block_test_time(1)<trial_time(:,1) & trial_time(:,1)<walking_block_test_time(2));
current_trial_time_eeg = trial_time(temp_trial_time_idx,:);
current_trial_time = current_trial_time_eeg-walking_block_test_time(1);
% save the data
filename = sub_dir + "\Partitioned_session_walking_block_test.mat";
save(filename,'eeg_data_all','goniometer_data_all','current_trial_time');

filename = sub_dir + "\_Partitioned_session_walking_block_test.mat";
save(filename,'eeg_data_all');

%% preprocess
base = sub_dir;
block_list = {'1', '2', '3', 'left', 'right', 'left_2sand'};
for i = 1:length(block_list)
    block = block_list{i};

    block_file = [base, '_Partitioned_session_walking_block_',block,'.mat'];

    output_location = base;
    location_file = [base, 'locations.ced'];
    save_dir = [base, 'eeg\'];
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    EEG = pop_importdata('dataformat','matlab','nbchan',0,'data',block_file,'setname','1','srate',1000,'pnts',0,'xmin',0);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'gui','off'); 
    EEG = eeg_checkset( EEG );
    EEG=pop_chanedit(EEG, 'load',{location_file,'filetype','autodetect'});
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.1,'hicutoff',50,'plotfreqz',1);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
    EEG = eeg_checkset( EEG );
    EEG = pop_reref( EEG, []);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'gui','off'); 
    EEG = eeg_checkset( EEG );
    EEG = pop_resample( EEG, 100);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'gui','off'); 
    EEG = eeg_checkset( EEG );
    %EEG = pop_runica(EEG, 'icatype', 'sobi');
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');%
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    out = 'test.txt';
    %EEG.etc.eeglabvers = '2023.0'; % this tracks which version of EEGLAB is being used, you may ignore it
    EEG = pop_rmbase( EEG, [],[]);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'gui','off'); 
    %eeglab redraw;
    tmp = EEG;
    epochLength = 5;

    epochSamples = epochLength ;

    EEG_epoch = eeg_regepochs(EEG, 'recurrence', epochSamples);

    size(EEG_epoch.data)
    [art, horiz, vert, blink, disc,...
            soglia_DV, diff_var, soglia_K, med2_K, meanK, soglia_SED, med2_SED, SED, soglia_SAD, med2_SAD, SAD, ...
            soglia_GDSF, med2_GDSF, GDSF, soglia_V, med2_V, nuovaV, soglia_D, maxdin]=ADJUST (EEG_epoch,out);

    indexList = [];
    tmp_SAD = SAD ./ soglia_SAD;
    tmp_SED = SED./soglia_SED;
    tmp_GDSF = GDSF./soglia_GDSF;
    tmp_V = nuovaV./soglia_V;
    tmp_K = meanK./soglia_K;
    for index = 1:15
        if (processValues(SAD(index) / soglia_SAD) + processValues(SED(index)/soglia_SED) + processValues(GDSF(index)/soglia_GDSF) + processValues(nuovaV(index)/soglia_V) + processValues(meanK(index)/soglia_K)) > 4

            indexList = [indexList, index];
        elseif processValues(SAD(index) / soglia_SAD) > 2
            indexList = [indexList, index];
        elseif processValues(SED(index)/soglia_SED) > 2
            indexList = [indexList, index];
        elseif processValues(GDSF(index)/soglia_GDSF) > 2
            indexList = [indexList, index];
        elseif processValues(nuovaV(index)/soglia_V) > 2
            indexList = [indexList, index];
        elseif processValues(meanK(index)/soglia_K) > 2
            indexList = [indexList, index];
        end
    end

    disp(indexList)
    EEG = pop_subcomp( EEG, indexList, 0);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4,'gui','off'); 
    EEG = pop_select( EEG, 'rmchannel',{'FT9','TP9','TP10','FT10'});
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 5,'gui','off'); 
    pop_saveh( ALLCOM, 'eeglabhist.m', 'C:\Users\fuxi0010.STUDENT\Desktop\');
    save_file = [];
    setFileName = ['blk_',block,'.set'];
    filePath = [base, 'eeg\'];
    EEG = pop_saveset( EEG, 'filename',setFileName,'filepath',filePath);
    
    data = pop_loadset(setFileName, filePath);
    load([base, '\Partitioned_session_walking_block_', block, '.mat']);
    eeg_data_all = data.data;
    save([base, 'eeg\blk_',block,'.mat'], 'current_trial_time', 'goniometer_data_all', 'eeg_data_all');
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    %eeglab redraw;
end

%% partition into trials
base = [base, '\eeg\'];
dsamp = 10;
a = 0;
%% walking block 1
load([base, '\blk_1.mat']);
new_folder_name = [base, '\walking_block1'];
goniometer_data_all = downsample(goniometer_data_all, dsamp);
goniometer_data_all = goniometer_data_all(:, 1:6);
mkdir(new_folder_name);
flag = 1;
if size(current_trial_time,1) == 40
    for idx_trial  = flag: flag + size(current_trial_time,1)-1
        %get the EEG trials in this session and save it
        trial_start_time = current_trial_time(idx_trial,1)/dsamp;
        trial_end_time = (current_trial_time(idx_trial,2)-1)/dsamp;
        eeg_data = eeg_data_all(:,trial_start_time:trial_end_time)';

        % get the goniometer trials in this session and save it
        goniometer_data = goniometer_data_all(trial_start_time:trial_end_time,:);

        filename = join([new_folder_name,'\trial',num2str(idx_trial),'.mat']);
        save(filename,'eeg_data','goniometer_data');
    end
end
%% walking block 2
%flag = flag + size(current_trial_time,1);
load([base, '\blk_2.mat']);
new_folder_name = [base, '\walking_block2'];
goniometer_data_all = downsample(goniometer_data_all, dsamp);
goniometer_data_all = goniometer_data_all(:, 1:6);
mkdir(new_folder_name);
for idx_trial  = flag: flag + size(current_trial_time,1)-1
    %get the EEG trials in this session and save it
    trial_start_time = current_trial_time(idx_trial,1)/dsamp;
    trial_end_time = (current_trial_time(idx_trial,2)-1)/dsamp;
    eeg_data = eeg_data_all(:,trial_start_time:trial_end_time)';

    % get the goniometer trials in this session and save it
    goniometer_data = goniometer_data_all(trial_start_time:trial_end_time,:);
    filename = join([new_folder_name,'\trial',num2str(idx_trial + size(current_trial_time,1)),'.mat']);
    save(filename,'eeg_data','goniometer_data');
end

%% walking block 3
%flag = flag + size(current_trial_time,1);
load([base, '\blk_3.mat']);
new_folder_name = [base, '\walking_block3'];
goniometer_data_all = downsample(goniometer_data_all, dsamp);
goniometer_data_all = goniometer_data_all(:, 1:6);
mkdir(new_folder_name);
for idx_trial  = flag: flag + size(current_trial_time,1)-1
    %get the EEG trials in this session and save it
    trial_start_time = current_trial_time(idx_trial,1)/dsamp;
    trial_end_time = (current_trial_time(idx_trial,2)-1)/dsamp;
    eeg_data = eeg_data_all(:,trial_start_time:trial_end_time)';

    % get the goniometer trials in this session and save it
    goniometer_data = goniometer_data_all(trial_start_time:trial_end_time,:);

    filename = join([new_folder_name,'\trial',num2str(idx_trial + 120),'.mat']);
    save(filename,'eeg_data','goniometer_data');
end

%% walking abnormal left
load([base, 'blk_left.mat']);
new_folder_name = [base, '\walking_abnormal_left'];
goniometer_data_all = downsample(goniometer_data_all, dsamp);
goniometer_data_all = goniometer_data_all(:, 1:6);
mkdir(new_folder_name);
for idx_trial  = 1: size(current_trial_time,1)
    %get the EEG trials in this session and save it
    trial_start_time = current_trial_time(idx_trial,1)/dsamp;
    trial_end_time = (current_trial_time(idx_trial,2)-1)/dsamp;
    eeg_data = eeg_data_all(:,trial_start_time:trial_end_time)';

    % get the goniometer trials in this session and save it
    goniometer_data = goniometer_data_all(trial_start_time:trial_end_time,:);

    filename = join([new_folder_name,'\trial',num2str(idx_trial + 80),'.mat']);
    save(filename,'eeg_data','goniometer_data');
end

%% walking abnormal right
load([base, 'blk_right.mat']);
new_folder_name = [base, '\walking_abnormal_right'];
goniometer_data_all = downsample(goniometer_data_all, dsamp);
goniometer_data_all = goniometer_data_all(:, 1:6);
mkdir(new_folder_name);
for idx_trial  = 1: size(current_trial_time,1)
    %get the EEG trials in this session and save it
    trial_start_time = current_trial_time(idx_trial,1)/dsamp;
    trial_end_time = (current_trial_time(idx_trial,2)-1)/dsamp;
    eeg_data = eeg_data_all(:,trial_start_time:trial_end_time)';

    % get the goniometer trials in this session and save it
    goniometer_data = goniometer_data_all(trial_start_time:trial_end_time,:);

    filename = join([new_folder_name,'\trial',num2str(idx_trial + 100),'.mat']);
    save(filename,'eeg_data','goniometer_data');
end




function result = processValues(input)
    result = input;
    result(input < 0) = 0;
end
