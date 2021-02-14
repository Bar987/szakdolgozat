from patient_v2 import Patient
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from lhyp_dataset_v2 import LhypDataset
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import copy
from sklearn.metrics import f1_score,precision_recall_fscore_support, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import scale
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from cnn_model import CNN_Model
from rnn_model import RNN_Model
from kornia.losses.focal import FocalLoss
from collections import Counter
import json
import os
from PIL import Image
from build_dataset_v2 import draw_images
import sys
from functools import partial
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.ax import AxSearch

DATA_PATH = 'drive/data/pickle/'
MODEL_PATH = 'drive/models/'
RUNS_PATH = 'drive/My Drive/fontos_egyetem/szakdoga/data/run/'

torch.manual_seed(42)
np.random.seed(42)

def training(cropped = '', with_RNN = True, model_name = 'resnet', pretrained = True, train_weights = 1, cnn_dropout = 0, rnn_type='gru',rnn_layer_num = 2, rnn_dropout = 0, optim_param = 'adam', lr_schedule = True,
     num_of_classes = 3, seq_len = 13, embedding_dim = 32, hidden_dim  = 32, batch_size = 16, num_of_epochs = 100, learning_rate = 1e-04, loss_fn = 'focal', checkpoint_dir=None):
    lr_list = []
    losses = []

    origin_model = ''

    dataset, val_dataset = load_data()

    class_sample_count = [0]*num_of_classes
    for sample in dataset:
        class_sample_count[sample[1]] += 1

    weights = []
    for data in dataset:
        weights.append(1/class_sample_count[data[1]])
    sampler = WeightedRandomSampler(weights, len(weights))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler = sampler)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,  shuffle=False)


    cnn_model = CNN_Model(model_name, embedding_dim, isFullNet = not with_RNN, num_of_class=num_of_classes, pretrained=pretrained, train_weights=train_weights, dropout = cnn_dropout)

    rnn_model = RNN_Model(rnn_type, embedding_dim, num_of_classes, rnn_layer_num, hidden_dim, rnn_dropout)

    

    max_val_f1 = 0.0
    min_loss = 9999999999999999999.0
    last_saved_loss = 0
    
    best_model_wts_cnn = copy.deepcopy(cnn_model.state_dict())
    best_model_wts_rnn = copy.deepcopy(rnn_model.state_dict())

    if torch.cuda.is_available():
        cnn_model = cnn_model.cuda()
        rnn_model = rnn_model.cuda()

    
    if loss_fn == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    elif loss_fn == 'weighted_cross':
        criterion = nn.CrossEntropyLoss((1/torch.Tensor(class_sample_count)).cuda())
    else :
        criterion = nn.CrossEntropyLoss()
    

    scheduler = None
    if optim_param == 'adam':
        optimizer = optim.Adam(
            list(cnn_model.parameters()) + list(rnn_model.parameters()), lr=learning_rate, weight_decay=1e-3)
    if optim_param == 'amsgrad':
        optimizer = optim.AdamW(list(cnn_model.parameters()) + list(rnn_model.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, amsgrad=True)
    elif optim_param == 'rmsprop':
        optimizer = optim.RMSprop(
            list(cnn_model.parameters()) + list(rnn_model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=1e-3)
    else:
        optimizer = optim.SGD(
            list(cnn_model.parameters()) + list(rnn_model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=1e-3, nesterov=True)
        
    base_lr = 0
    max_lr = 0
    if lr_schedule:
        base_lr = 1e-6
        max_lr = 1e-4
        if with_RNN:
            step = 300
        else:
            step = 3000
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr
                , step_size_up=step, step_size_down=step, mode='exp_range'
                , gamma=0.999, scale_fn=None, scale_mode='cycle', cycle_momentum=False
                , base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

    if checkpoint_dir:
        cnn_model_state, rnn_model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        cnn_model.load_state_dict(cnn_model_state)
        rnn_model.load_state_dict(rnn_model_state)
        optimizer.load_state_dict(optimizer_state)

    all_val_errors = []

    for iteration in range(num_of_epochs):
        #
        # Training
        #
        train_loss, train_acc = train(cnn_model, with_RNN, rnn_model, dataloader, criterion, optimizer, scheduler)
        print(iteration,' Train loss: ', train_loss, ' Train acc ', train_acc)
        #
        # Validating
        #
        val_running_loss, val_acc, val_f1, val_errors, mcc = validate(cnn_model, with_RNN, rnn_model, val_dataloader, criterion)

        print(iteration,' Val loss: ', val_running_loss,
              ' Val acc ', val_acc,
              'F1:', val_f1,
              'MCC', mcc)

        with tune.checkpoint_dir(iteration) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((cnn_model.state_dict(), rnn_model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(loss=val_running_loss, accuracy=val_acc, f1_score=val_f1)


def train(cnn_model, with_RNN, rnn_model, dataloader, criterion, optimizer, scheduler = None):
    running_loss = 0
    correct = 0.00
    total = 0.00

    cnn_model.train()
    rnn_model.train()
    for i, input_datas in enumerate(dataloader, 0):
        datas, labels, file_names = input_datas

        if torch.cuda.is_available():
            datas = datas.cuda().float()
            labels = labels.cuda()

        optimizer.zero_grad()

        if with_RNN:
            outputs = rnn_model(cnn_model(datas).cuda())
        else:
            outputs = cnn_model(datas)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if scheduler != None:
            scheduler.step()
        

    return running_loss, correct/total


def validate(cnn_model, with_RNN, rnn_model, val_dataloader, criterion):
    val_running_loss = 0
    val_correct = 0.00
    val_total = 0.00
    y_true = np.array([])
    y_pred = np.array([])

    errors = []
    cnn_model.eval()
    rnn_model.eval()
    for i, val_input_datas in enumerate(val_dataloader, 0):
        val_datas, val_labels, val_file_names = val_input_datas

        y_true = np.concatenate((y_true, val_labels.numpy()), 0)
        if torch.cuda.is_available():
            val_datas = val_datas.cuda()
            val_labels = val_labels.cuda()

        if with_RNN:
            val_outputs = rnn_model(cnn_model(val_datas).cuda())
        else:
            val_outputs = cnn_model(val_datas)

        val_loss = criterion(val_outputs, val_labels)

        val_running_loss += val_loss.item()
        _, val_predicted = torch.max(val_outputs, 1)

        y_pred = np.concatenate((y_pred, val_predicted.cpu().numpy()), 0)

        val_total += val_labels.size(0)

        batch_err = [ j for j, x in enumerate(val_predicted.eq(val_labels)) if x == False]
        errors += [val_file_names[i] for i in batch_err]

        val_correct += val_predicted.eq(val_labels).sum().item()

        pres, recall, f1, supp= precision_recall_fscore_support(y_true, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)

    return val_running_loss, val_correct/val_total, f1, errors, mcc


def test(cnn_model, with_RNN, rnn_model, test_dataloader, criterion):
    val_running_loss = 0
    val_correct = 0.00
    val_total = 0.00
    y_true = np.array([])
    y_pred = np.array([])

    errors = []

    cnn_model.eval()
    rnn_model.eval()

    for i, test_input_datas in enumerate(test_dataloader, 0):
        val_datas, val_labels, val_file_names = test_input_datas
        if torch.cuda.is_available():
            val_datas = val_datas.cuda()
            val_labels = val_labels.cuda()

        y_true = np.concatenate((y_true, val_labels.cpu().numpy()), 0)
        
        if with_RNN:
            val_outputs = rnn_model(cnn_model(val_datas).cuda())
        else:
            val_outputs = cnn_model(val_datas)

        val_loss = criterion(val_outputs, val_labels)

        val_running_loss += val_loss.item()
        _, val_predicted = torch.max(val_outputs, 1)

        y_pred = np.concatenate((y_pred, val_predicted.cpu().numpy()), 0)
        val_total += val_labels.size(0)

        
        val_correct += val_predicted.eq(val_labels).sum().item()

        batch_err = [ j for j, x in enumerate(val_predicted.eq(val_labels)) if x == False]
        errors += [val_file_names[i] for i in batch_err]

    print('Errors: ' ,errors)
    
    test_loss = val_running_loss
    test_acc = val_correct/val_total
    pres, recall, f1, supp= precision_recall_fscore_support(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    print('Test loss: ', test_loss, ' Test acc ', test_acc)
    print('Pres:  ', pres, 'Recall:  ', recall, 'F1 score: ', f1, 'Supp :', supp, 'MCC :', mcc)
    return confusion_matrix(y_true, y_pred, normalize='pred'), test_loss, test_acc, f1, mcc

def evaluate(model_pth):

    print(model_pth)
    origin_model = model_pth

    prop = model_pth.split('_')

    model_name = prop[0]
    num_of_classes = int(prop[1])
    seq_len = int(prop[2])
    

    with open(MODEL_PATH + origin_model + '/prop.txt', 'r') as json_file:
        data = json.load(json_file)

    batch_size = 16
    
    embedding_dim = data['embedding_dim']
    hidden_dim = data['hidden_dim']

    loss_fn = data['loss_fn']
    rnn_type = data['rnn']
    with_Rnn = data['with_rr']
    mode = data['mode']

    with open(DATA_PATH+mode+'val_'+str(num_of_classes)+'_'+str(seq_len), 'rb') as infile:
        val_pat = pickle.load(infile, encoding='bytes')

    with open(DATA_PATH+mode+'test_'+str(num_of_classes)+'_'+str(seq_len), 'rb') as infile:
        test_pat = pickle.load(infile, encoding='bytes')

    mean, std = calculate_mean_std(test_pat+val_pat)

    val_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128), interpolation = Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([mean, mean, mean], [std, std, std])
    ])

    val_dataset = LhypDataset(
        val_pat, val_trans, False)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = LhypDataset(
        test_pat, val_trans, False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)


    cnn_model = CNN_Model(model_name, embedding_dim)

    rnn_model = RNN_Model(rnn_type, embedding_dim, num_of_classes, 2, hidden_dim, 0)

    max_val_acc = 0.0
    min_loss = 9999999999999999999.0
    last_saved_loss = 0

    if origin_model != '':
        cnn_model.load_state_dict(torch.load(MODEL_PATH+ origin_model+'/cnn.pth'))
        rnn_model.load_state_dict(torch.load(MODEL_PATH+ origin_model +'/rnn.pth'))

    best_model_wts_cnn = copy.deepcopy(cnn_model.state_dict())
    best_model_wts_rnn = copy.deepcopy(rnn_model.state_dict())

    if torch.cuda.is_available():
        cnn_model = cnn_model.cuda()
        rnn_model = rnn_model.cuda()

    if loss_fn == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    elif loss_fn == 'weighted_cross':
        criterion = nn.CrossEntropyLoss()
    else :
        criterion = nn.CrossEntropyLoss()

    test(cnn_model, with_Rnn, rnn_model, val_dataloader, criterion)
    conf_mat, test_loss, test_acc, f1, mcc = test(cnn_model, with_Rnn, rnn_model, test_dataloader, criterion)

    if num_of_classes == 2:
        classes = {0: 'Normal', 1: 'Other'}
    elif num_of_classes == 3:
        classes = {0: 'Normal', 1: 'HCM', 2: 'Other'}

    df_cm = pd.DataFrame(conf_mat, index=[i for i in classes.values()],
                         columns=[i for i in classes.values()])
    plt.figure(2,figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def calculate_mean_std(all_data):
    mean = torch.zeros(1)
    std = torch.zeros(1)

    for patient in all_data:
        for img in patient['img']:
            cucc = img / 255.00
            mean += np.mean(cucc)
            std += np.std(cucc)


    mean = np.divide(mean, len(all_data)* len(all_data[0]))
    std = np.divide(std, len(all_data)* len(all_data[0]))

    return mean, std

def filter_out_patients(patients, filtered_out):
    temp = []
    filtered = []
    for patient in patients:
        if not patient['filename'] in filtered_out:
            temp.append(patient)
        else:
            filtered.append(patient['label'])
    print(filtered)
    return temp

def load_data(data_dir=None):

    with open(MODEL_PATH+'/croppedtrain_3_13', 'rb') as infile:
        train_pat = pickle.load(infile, encoding='bytes')
    with open(MODEL_PATH+'/croppedval_3_13', 'rb') as infile:
        val_pat = pickle.load(infile, encoding='bytes')

    all_data = train_pat + val_pat

    mean, std = calculate_mean_std(all_data)
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150 , 150), interpolation = Image.BILINEAR),
        transforms.RandomRotation((60)),
        transforms.RandomCrop((128, 128), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize([mean, mean, mean], [std, std, std])
    ])

    val_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128), interpolation = Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([mean, mean, mean], [std, std, std])
    ])

    
    dataset = LhypDataset(train_pat, trans, True , True)

    val_dataset = LhypDataset(
        val_pat, val_trans, False, True)

    return dataset, val_dataset


def main(argv):
    if argv[0] == 'eval':
        evaluate(argv[1])
    else:

        tune_net(num_samples=50, max_num_epochs=20)
    


def train_tune(config, checkpoint_dir=None):
    training(cropped = 'cropped', with_RNN = True, model_name = 'densenet', pretrained=True, train_weights=config['train_weights']
    , cnn_dropout=config['cnn_dropout'], rnn_type =config['rnn_type'], rnn_dropout=config['rnn_dropout'], rnn_layer_num=config['rnn_layer_num']
    , optim_param = config['optim_param'], lr_schedule = False, num_of_classes = 3, seq_len = 13, embedding_dim = config['embedding_dim']
    , hidden_dim = config['hidden_dim'], batch_size = config['batch_size'], num_of_epochs = 20, learning_rate = config['learning_rate'], loss_fn = 'cross',  checkpoint_dir=checkpoint_dir)

def tune_net(num_samples=10, max_num_epochs=25):
    data_dir = os.path.abspath("./data")
    config = {
        "train_weights": tune.choice([float(1.0), 0.8, 0.6]),
        "rnn_type": tune.choice(['gru', 'lstm']),
        "optim_param": tune.choice(['adam', 'rmsprop', 'amsgrad']),
        "cnn_dropout": tune.choice([0.3, 0.5, 0.7]),
        "rnn_dropout": tune.choice([0.3, 0.5, 0.7]),
        "rnn_layer_num": tune.choice([2 , 3]),
        "embedding_dim": tune.choice([32, 64, 128]),
        "hidden_dim": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
    }


    scheduler = ASHAScheduler(
        metric="f1_score",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=4)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", 'f1_score', "training_iteration"])

    algo = AxSearch(
        metric ="f1_score",
        mode="max"
    )
    
    result = tune.run(
        partial(train_tune, checkpoint_dir='tune/'),
        resources_per_trial={"cpu": 2, 'gpu': 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        search_alg=algo,
        keep_checkpoints_num = 1,
        checkpoint_score_attr = 'f1_score'
        )

    best_trial = result.get_best_trial("f1_score", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    cnn_model = CNN_Model('densenet', best_trial.config["embedding_dim"], isFullNet = False, num_of_class=3, pretrained=True, train_weights=best_trial.config["train_weights"])
    rnn_model = RNN_Model(best_trial.config["rnn_type"], best_trial.config["embedding_dim"], 3, best_trial.config["rnn_layer_num"], best_trial.config["hidden_dim"], 0.5)
    
    if torch.cuda.is_available():
        cnn_model = cnn_model.cuda()
        rnn_model = rnn_model.cuda()

    best_checkpoint_dir = result.get_best_checkpoint(best_trial, metric='f1_score', mode='max')
    cnn_model_state, rnn_model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    cnn_model.load_state_dict(cnn_model_state)
    rnn_model.load_state_dict(rnn_model_state)


    with open(MODEL_PATH+'/croppedtest_3_13', 'rb') as infile:
        test_pat = pickle.load(infile, encoding='bytes')

    mean, std = calculate_mean_std(test_pat)

    val_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128), interpolation = Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([mean, mean, mean], [std, std, std])
    ])


    test_dataset = LhypDataset(
        test_pat, val_trans, False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False)

    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    
    conf_mat, test_loss, test_acc, f1, mcc = test(cnn_model, True, rnn_model, test_dataloader, criterion)
    print("Best trial test set f1: {}".format(f1))

if __name__ == "__main__":
    main(sys.argv[1:])

