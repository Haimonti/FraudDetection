#!/usr/bin/env python
import numpy as np
import json
import pandas as pd
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data.data import AAERComp #,LibSVMData, LibCSVData, CriteoCSVData
# from data.sparse_data import LibSVMDataSp
from models.mlp import MLP_1HL, MLP_2HL, MLP_3HL
from models.dynamic_net import DynamicNet, ForwardType
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import SGD, Adam
from misc.auc import auc

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE


parser = argparse.ArgumentParser()
parser.add_argument('--feat_d', type=int, required=True)
parser.add_argument('--hidden_d', type=int, required=True)
parser.add_argument('--boost_rate', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--num_nets', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--tr', type=str, required=True)
parser.add_argument('--te', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs_per_stage', type=int, required=True)
parser.add_argument('--correct_epoch', type=int ,required=True)
parser.add_argument('--L2', type=float, required=True)
parser.add_argument('--sparse', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--normalization', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--cv', default=False, type=lambda x: (str(x).lower() == 'true')) 
parser.add_argument('--model_order',default='second', type=str)
parser.add_argument('--out_f', type=str, required=True)
parser.add_argument('--cuda', action='store_true')

opt = parser.parse_args()

if not opt.cuda:
    torch.set_num_threads(16)

# prepare the dataset
def get_data():
   
    data = pd.read_csv('data_/merged_compustat_and_labels.csv')
    with open('data_/features.json') as json_file:
        data_items = json.load(json_file)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    data = data.fillna(0)
    train_period = (1990,2002)
    test_period = (2003,2019)
    train = data[(data['fyear'] >= train_period[0]) & (data['fyear'] <= train_period[1])]
    test = data[(data['fyear'] >= test_period[0]) & (data['fyear'] <= test_period[1])]
    val =[]
    if opt.feat_d == 42:
      feature = 'features'
    elif opt.feat_d == 28:
      feature = 'raw_financial_items_28'
    elif opt.feat_d == 14:
      feature = 'financial_ratios_14'

    X_train = train[data_items[feature]]
    y_train = train['misstate'] #

    X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

    train = pd.concat([X_train, y_train], axis=1)
    
    test = pd.concat([test[data_items[feature]],test['misstate']], axis=1) #

    train_ds = AAERComp(train, 'misstate')
    test_ds = AAERComp(test, 'misstate')
    print(f'#Train: {len(train)}, #Val: {len(val)} #Test: {len(test)}')
    return train_ds, test_ds, val


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    return optimizer

def accuracy(net_ensemble, test_loader):
    correct = 0
    total = 0
    loss = 0
    for x, y in test_loader:
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            middle_feat, out = net_ensemble.forward(x)
        correct += (torch.sum(y[out > 0.] > 0) + torch.sum(y[out < .0] < 0)).item()
        total += y.numel()
    return correct / total

def logloss(net_ensemble, test_loader):
    loss = 0
    total = 0
    loss_f = nn.BCEWithLogitsLoss() # Binary cross entopy loss with logits, reduction=mean by default
    for x, y in test_loader:
        if opt.cuda:
            x, y= x.cuda(), y.cuda().view(-1, 1)
        y = (y + 1) / 2
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
        loss += loss_f(out, y)
        total += 1

    return loss / total

def auc_score(net_ensemble, test_loader):
    actual = []
    posterior = []
    for x, y in test_loader:
        if opt.cuda:
            x = x.cuda()
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        prob = 1.0 - 1.0 / torch.exp(out)   # Why not using the scores themselve than converting to prob
        prob = prob.cpu().numpy().tolist()
        posterior.extend(prob)
        actual.extend(y.numpy().tolist())
    score = auc(actual, posterior)
    return score

def init_gbnn(train):
    positive = negative = 0
    print(train)

    for i in range(len(train)):
        if train[i][1] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'Blind accuracy: {blind_acc}')
    #print(f'Blind Logloss: {blind_acc}')
    return float(np.log(positive / negative))

if __name__ == "__main__":

    train, test, val = get_data()
    
    print(opt.data + ' training and test datasets are loaded!')
    train_loader = DataLoader(train, opt.batch_size, shuffle = True, drop_last=False, num_workers=2)
    test_loader = DataLoader(test, opt.batch_size, shuffle=False, drop_last=False, num_workers=2)
    if opt.cv:
        val_loader = DataLoader(val, opt.batch_size, shuffle=True, drop_last=False, num_workers=2)
    # For CV use
    best_score = 0
    val_score = best_score
    best_stage = opt.num_nets-1

    c0 = init_gbnn(train)
    net_ensemble = DynamicNet(c0, opt.boost_rate)
    loss_f1 = nn.MSELoss(reduction='none')
    loss_f2 = nn.BCEWithLogitsLoss(reduction='none')
    loss_models = torch.zeros((opt.num_nets, 3))

    all_ensm_losses = []
    all_ensm_losses_te = []
    all_mdl_losses = []
    dynamic_br = []


    for stage in range(opt.num_nets):
        t0 = time.time()
        model = MLP_2HL.get_model(stage, opt)  # Initialize the model_k: f_k(x), multilayer perception v2
        if opt.cuda:
            model.cuda()

        optimizer = get_optim(model.parameters(), opt.lr, opt.L2)
        net_ensemble.to_train() # Set the models in ensemble net to train mode

        stage_mdlloss = []
        for epoch in range(opt.epochs_per_stage):
            for i, (x, y) in enumerate(train_loader):
                if opt.cuda:
                    x, y= x.cuda(), y.cuda().view(-1, 1)
                middle_feat, out = net_ensemble.forward(x)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                if opt.model_order=='first':
                    grad_direction = y / (1.0 + torch.exp(y * out))
                else:
                    h = 1/((1+torch.exp(y*out))*(1+torch.exp(-y*out)))
                    grad_direction = y * (1.0 + torch.exp(-y * out))
                    out = torch.as_tensor(out)
                    nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                _, out = model(x, middle_feat)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate*out, grad_direction)  # T
                loss = loss*h
                loss = loss.mean()
                model.zero_grad()
                loss.backward()
                optimizer.step()
                stage_mdlloss.append(loss.item()) 

        net_ensemble.add(model)
        sml = np.mean(stage_mdlloss)


        stage_loss = []
        lr_scaler = 2
        # fully-corrective step
        if stage != 0:
            # Adjusting corrective step learning rate 
            if stage % 15 == 0:
                #lr_scaler *= 2
                opt.lr /= 2
                opt.L2 /= 2
            optimizer = get_optim(net_ensemble.parameters(), opt.lr / lr_scaler, opt.L2)
            for _ in range(opt.correct_epoch):
                for i, (x, y) in enumerate(train_loader):
                    if opt.cuda:
                        x, y = x.cuda(), y.cuda().view(-1, 1)
                    _, out = net_ensemble.forward_grad(x)
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    y = (y + 1.0) / 2.0
                    loss = loss_f2(out, y).mean() 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_loss.append(loss.item())

        
        sl_te = logloss(net_ensemble, test_loader)
        # Store dynamic boost rate
        dynamic_br.append(net_ensemble.boost_rate.item())
        # store model
        net_ensemble.to_file(opt.out_f)
        net_ensemble = DynamicNet.from_file(opt.out_f, lambda stage: MLP_2HL.get_model(stage, opt))

        elapsed_tr = time.time()-t0
        sl = 0
        if stage_loss != []:
            sl = np.mean(stage_loss)

        

        all_ensm_losses.append(sl)
        all_ensm_losses_te.append(sl_te.item())
        all_mdl_losses.append(sml)



        print(f'Stage - {stage}, training time: {elapsed_tr: .1f} sec, boost rate: {net_ensemble.boost_rate.item(): .4f}, Training Loss: {sl: .4f}, Test Loss: {sl_te: .4f}')

        if opt.cuda:
            net_ensemble.to_cuda()
        net_ensemble.to_eval() # Set the models in ensemble net to eval mode

        # Train
        print('Acc results from stage := ' + str(stage) + '\n')
        # AUC
        if opt.cv:
            val_score = auc_score(net_ensemble, val_loader) 
            if val_score > best_score:
                best_score = val_score
                best_stage = stage

        test_score = auc_score(net_ensemble, test_loader)
        print(f'Stage: {stage}, AUC@Val: {val_score:.4f}, AUC@Test: {test_score:.4f}')

        loss_models[stage, 1], loss_models[stage, 2] = val_score, test_score

    val_auc, te_auc = loss_models[best_stage, 1], loss_models[best_stage, 2]
    print(f'Best validation stage: {best_stage},  AUC@Val: {val_auc:.4f}, final AUC@Test: {te_auc:.4f}')

    loss_models = loss_models.detach().cpu().numpy()
    fname = 'tr_ts_raw' + opt.data +'_auc'
    np.save(fname, loss_models) 
    fname = './results/' + opt.data + '_cls_raw'
    np.savez(fname, training_loss=all_ensm_losses, test_loss=all_ensm_losses_te, model_losses=all_mdl_losses, dynamic_boostrate=dynamic_br, params=opt)

