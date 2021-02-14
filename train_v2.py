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

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = 'drive/data/pickle/'
MODEL_PATH = 'drive/models/'
RUNS_PATH = 'drive/My Drive/fontos_egyetem/szakdoga/data/run/'

def training(cropped = '', with_RNN = True, model_name = 'resnet', pretrained = True, train_weights = 1, cnn_dropout = 0, rnn_type='gru', rnn_dropout = 0, bidirectional=False, optim_param = 'adam', lr_schedule = True,
     num_of_classes = 3, seq_len = 13, embedding_dim = 32, hidden_dim  = 32, batch_size = 16, num_of_epochs = 100, learning_rate = 1e-04, loss_fn = 'focal'):
    lr_list = []
    losses = []

    origin_model = ''

    filtered_out = []

    with open(DATA_PATH+cropped+'train_'+str(num_of_classes)+'_'+str(seq_len), 'rb') as infile:
        train_pat = filter_out_patients(pickle.load(infile, encoding='bytes'), filtered_out)
    with open(DATA_PATH+cropped+'val_'+str(num_of_classes)+'_'+str(seq_len), 'rb') as infile:
        val_pat = filter_out_patients(pickle.load(infile, encoding='bytes'), filtered_out)
    with open(DATA_PATH+cropped+'test_'+str(num_of_classes)+'_'+str(seq_len), 'rb') as infile:
        test_pat = filter_out_patients(pickle.load(infile, encoding='bytes'), filtered_out)


    all_data = train_pat + val_pat + test_pat

    mean, std = calculate_mean_std(all_data)

    model_id = model_name +'_'+ str(num_of_classes)+'_'+ str(seq_len) +'_'+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.mkdir(MODEL_PATH + model_id) 
    print(model_id)

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

    
    dataset = LhypDataset(train_pat, trans, True , with_RNN)

    test_dataset = LhypDataset(
        test_pat, val_trans, False, with_RNN)

    val_dataset = LhypDataset(
        val_pat, val_trans, False, with_RNN)

    print(len(dataset), len(val_dataset), len(test_dataset))

    class_sample_count = [0]*num_of_classes
    for sample in dataset:
        class_sample_count[sample[1]] += 1
    print(class_sample_count)

    weights = []
    for data in dataset:
        weights.append(1/class_sample_count[data[1]])
    sampler = WeightedRandomSampler(weights, len(weights))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler = sampler)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,  shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)


    cnn_model = CNN_Model(model_name, embedding_dim, isFullNet = not with_RNN, num_of_class=num_of_classes, pretrained=pretrained, train_weights=train_weights, dropout = cnn_dropout)

    rnn_model = RNN_Model(rnn_type, embedding_dim, num_of_classes, 2, hidden_dim, rnn_dropout, bidirectional)

    max_val_f1 = 0.0
    min_loss = 9999999999999999999.0
    last_saved_loss = 0

    if origin_model != '':
        cnn_model.load_state_dict(torch.load(MODEL_PATH+ origin_model+'/cnn.pth'))
        rnn_model.load_state_dict(torch.load(MODEL_PATH+ origin_model +'/rnn.pth'))
        
        with open(MODEL_PATH+ origin_model + '/prop.txt', 'r') as json_file:
            data = json.load(json_file)
            min_loss = data['min_val_loss']
    
    best_model_wts_cnn = copy.deepcopy(cnn_model.state_dict())
    best_model_wts_rnn = copy.deepcopy(rnn_model.state_dict())

    print(torch.cuda.is_available())
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
            list(cnn_model.parameters()) + list(rnn_model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=1e-3)

    if lr_schedule:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 8, T_mult=1, eta_min=1e-5, last_epoch=-1, verbose=True)

    patience_setting = 15
    patience = patience_setting

    writer = SummaryWriter(RUNS_PATH + model_id)


    properties = {
        'mode': cropped, 
        'name': model_name,
        'pretrained': pretrained,
        'kept_layers': train_weights,
        'with_rr': with_RNN,
        'rnn': rnn_type,
        'cnn_dropout': cnn_dropout,
        'rnn_dropout': rnn_dropout,
        'hidden_dim': hidden_dim,
        'origin': origin_model,
        'embedding_dim': embedding_dim,
        'loss_fn': loss_fn,
        'optim': optim_param,
        'lr': learning_rate,
        'lr_schedule': lr_schedule,
        'batch_size': batch_size,
        'patience': patience_setting,
        'iteration': 0,
        'min_val_loss': min_loss,
        'test_loss': min_loss,
        'test_acc': 0,
        'f1_score': 0
    }

    print(properties)
    with open(MODEL_PATH + model_id + '/prop.txt', 'w') as outfile:
        json.dump(properties, outfile)
    


    all_val_errors = []

    for iteration in range(num_of_epochs):
        print(iteration)
        #
        # Training
        #
        train_loss, train_acc = train(cnn_model, with_RNN, rnn_model, dataloader, criterion, optimizer, scheduler)

        writer.add_scalar('Training loss', train_loss, iteration)
        writer.add_scalar('Training accuracy', train_acc, iteration)
        print('Train loss: ', train_loss, ' Train acc ', train_acc)
        #
        # Validating
        #
        val_running_loss, val_acc, val_f1, val_errors, mcc = validate(cnn_model, with_RNN, rnn_model, val_dataloader, criterion)

        all_val_errors += val_errors

        writer.add_scalar('Validation loss', val_running_loss, iteration)
        writer.add_scalar('Validation acc', val_acc, iteration)
        writer.add_scalar('Validation f1',
                          val_f1, iteration)

        # scheduler.step()

        print('Val loss: ', val_running_loss,
              ' Val acc ', val_acc,
              'F1:', val_f1,
              'MCC', mcc)

        

        if val_running_loss >= min_loss:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break
        else:
            min_loss = val_running_loss
            patience = patience_setting


        if val_f1 > max_val_f1:
            max_val_f1 = val_f1
            last_saved_loss = val_running_loss
            best_model_wts_cnn = copy.deepcopy(cnn_model.state_dict())
            best_model_wts_rnn = copy.deepcopy(rnn_model.state_dict())
            torch.save(best_model_wts_cnn, MODEL_PATH+ model_id + '/cnn.pth')
            torch.save(best_model_wts_rnn, MODEL_PATH+ model_id + '/rnn.pth')

        if val_f1 == max_val_f1 and last_saved_loss > val_running_loss:
            last_saved_loss = val_running_loss
            best_model_wts_cnn = copy.deepcopy(cnn_model.state_dict())
            best_model_wts_rnn = copy.deepcopy(rnn_model.state_dict())
            torch.save(best_model_wts_cnn, MODEL_PATH+ model_id + '/cnn.pth')
            torch.save(best_model_wts_rnn, MODEL_PATH+ model_id + '/rnn.pth')

        print('lr: ', get_lr(optimizer))
        lr_list.append(get_lr(optimizer))
        losses.append(val_running_loss)

        with open(MODEL_PATH+ model_id + '/prop.txt', 'r') as json_file:
            data = json.load(json_file)
            data['iteration'] = iteration
            data['min_val_loss'] = min_loss

        with open(MODEL_PATH+ model_id + '/prop.txt', 'w') as json_file:
            json.dump(data, json_file)

    val_errors_count = dict(Counter(errors for errors in all_val_errors))
    print({k: v for k, v in sorted(val_errors_count.items(), key=lambda item: item[1], reverse=True)})
        
    plt.figure(1)
    plt.plot(lr_list, losses)
    plt.xscale('log')
    plt.title('lr_test')
    plt.show()

    # # Testing the model
    
    cnn_model.load_state_dict(best_model_wts_cnn)
    rnn_model.load_state_dict(best_model_wts_rnn)
    conf_mat, test_loss, test_acc, f1, mcc = test(cnn_model, with_RNN, rnn_model, test_dataloader, criterion)

    with open(MODEL_PATH+ model_id + '/prop.txt', 'r') as json_file:
            data = json.load(json_file)
            data['test_loss'] = test_loss
            data['test_acc'] = test_acc
            data['f1_score'] = f1
            data['mcc'] = mcc


    with open(MODEL_PATH+ model_id + '/prop.txt', 'w') as json_file:
            json.dump(data, json_file)

    if num_of_classes == 2:
        classes = {0: 'Normal', 1: 'Other'}
    elif num_of_classes == 3:
        classes = {0: 'Normal', 1: 'HCM', 2: 'Other'}

    df_cm = pd.DataFrame(conf_mat, index=[i for i in classes.values()],
                         columns=[i for i in classes.values()])
    plt.figure(2,figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(MODEL_PATH+ model_id + '/res.png')


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

        pres, recall, f1, supp= precision_recall_fscore_support(y_true, y_pred, beta=1, average='weighted')
        mcc = matthews_corrcoef(y_true, y_pred)
    print('Errors: ' ,errors)

    return val_running_loss, val_correct/val_total, f1, errors, mcc


def test(cnn_model, with_RNN, rnn_model, test_dataloader, criterion, normalize='pred'):
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
    return confusion_matrix(y_true, y_pred, normalize=normalize), test_loss, test_acc, f1, mcc

def evaluate(model_pth, normalize="pred"):

    print(model_pth)
    origin_model = model_pth

    prop = model_pth.split('_')

    model_name = prop[0]
    num_of_classes = int(prop[1])
    seq_len = int(prop[2])
    

    with open(MODEL_PATH+ origin_model + '/prop.txt', 'r') as json_file:
        data = json.load(json_file)
    
    batch_size = 8
    
    embedding_dim = data['embedding_dim']
    hidden_dim = data['hidden_dim']

    loss_fn = data['loss_fn']
    rnn_type = data['rnn']
    with_Rnn = data['with_rr']
    mode = data['mode']

    with open(DATA_PATH+mode+'train_'+str(num_of_classes)+'_'+str(seq_len), 'rb') as infile:
        train_pat = pickle.load(infile, encoding='bytes')

    with open(DATA_PATH+mode+'val_'+str(num_of_classes)+'_'+str(seq_len), 'rb') as infile:
        val_pat = pickle.load(infile, encoding='bytes')

    with open(DATA_PATH+mode+'test_'+str(num_of_classes)+'_'+str(seq_len), 'rb') as infile:
        test_pat = pickle.load(infile, encoding='bytes')

    mean, std = calculate_mean_std(train_pat+test_pat+val_pat)

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

    print(torch.cuda.is_available())
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
    conf_mat, test_loss, test_acc, f1, mcc = test(cnn_model, with_Rnn, rnn_model, test_dataloader, criterion, normalize=normalize)

    if num_of_classes == 2:
        classes = {0: 'Normal', 1: 'Other'}
    elif num_of_classes == 3:
        classes = {0: 'Normal', 1: 'HCM', 2: 'Other'}

    df_cm = pd.DataFrame(conf_mat, index=[i for i in classes.values()],
                         columns=[i for i in classes.values()])
    plt.figure(2,figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def main(argv):
    if argv[0] == 'eval':
        if len(argv) > 2:
            evaluate(argv[1], argv[2])
        else:
            evaluate(argv[1], None)
    else:
        training(cropped = 'cropped', with_RNN = True, model_name = 'resnet', pretrained=False, train_weights=1 , cnn_dropout = 0.3, rnn_type ="lstm"
        , rnn_dropout = 0.5 , bidirectional=False, optim_param = 'rmsprop', lr_schedule = False, num_of_classes = 3, seq_len = 13, embedding_dim = 128
        , hidden_dim = 64, batch_size = 16, num_of_epochs = 50, learning_rate = 5e-05, loss_fn = 'focal')

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

    print(mean, std)
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

if __name__ == "__main__":
    main(sys.argv[1:])

