# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:56:50 2022

@author: AdrienAntoinette
"""

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../../")
from model import TCN

#from TCN.poly_music.utils import data_generator
import numpy as np
import pickle
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Sequence Modeling - Motion Prediction')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='Nott',
                    help='the dataset to run (default: Nott)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
#parser.add_argument("--save-model-path",type=str,default="tcn",
 #       help="Path to store saved models")

###Adam default, trying sgd
###next learnign rate =not necessarily. Could just be that our models are tried at 1e-3

#iter 1 - baseline - quat -SGD - lr = 1e-3 = running
#iter 2 - quat-Adam- same learning rate
#ite 3 - quat-SGD - smaller ler - 1e-4
#start with those then we'll add 3 to 5 more iters
#iter 4 - quat - 


args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
input_size =  24#72 #88

#import pickle
#import numpy as np
#import torch

#with open('train.pkl', 'rb') as f:
with open('../../data_out/quat/train.pkl', 'rb') as f:
    data = pickle.load(f)
X_train1= data[0]
Y_train1 = data[1]

X_train= []
Y_train = []
for i in range(len(X_train1)):
    #a = torch.from_numpy(X_train1[i])
    a = torch.FloatTensor(X_train1[i])
    b = torch.FloatTensor(Y_train1[i])
    X_train.append(a)
    Y_train.append(b)
X_train = np.array(X_train)
Y_train = np.array(Y_train)


#with open('validation.pkl', 'rb') as f:
with open('../../data_out/quat/validation.pkl', 'rb') as f:
    data = pickle.load(f)
X_valid1= data[0]
Y_valid1= data[1]
X_valid= []
Y_valid=[]
for i in range(len(X_valid1)):
    #a = torch.from_numpy(X_valid1[i])
    a = torch.FloatTensor(X_valid1[i])
    b = torch.FloatTensor(Y_valid1[i])
    X_valid.append(a)
    Y_valid.append(b)
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

#with open('test.pkl', 'rb') as f:
with open('../../data_out/quat/test.pkl', 'rb') as f:
    data = pickle.load(f)
X_test1= data[0]
Y_test1=data[1]
X_test= []
Y_test=[]
for i in range(len(X_test1)):
    #a = torch.from_numpy(X_test1[i])
    a = torch.FloatTensor(X_test1[i])
    b = torch.FloatTensor(Y_test1[i])
    X_test.append(a)
    Y_test.append(b)
    
X_test = np.array(X_test)
Y_test = np.array(Y_test)

#X_train, X_valid, X_test = data_generator(args.data)

n_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout

model = TCN(input_size, input_size, n_channels, kernel_size, dropout=args.dropout)


if args.cuda:
    model.cuda()

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

import math
def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        #pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        #pred = np.swapaxes(predAll[s][:,:count_,:],0,1)
        pred = predAll[s]
        
        #target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        #target = np.swapaxes(targetAll[s][:,:count_,:],0,1)
        target = targetAll[s]
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                #sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
                sum_+=math.sqrt((pred[i,t] - target[i,t])**2+(pred[i,t] - target[i,t])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All

def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    #seq_len, _, _ = pred_traj.size()
    #loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = pred_traj_gt - pred_traj
    loss = loss**2
    N = pred_traj.shape[0]
    T = pred_traj.shape[1]
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)).sum(dim=0) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1)).sum(dim=0)
    if mode == 'sum':
        torch.sum(loss)
    elif mode == 'raw':
        loss
    ade = loss/(N*T)
    return ade


#def evaluate(X_data, name='Eval'):
def evaluate(X_data,Y_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    sum_all = 0
    with torch.no_grad():
        for idx in eval_idx_list:
            data_line = X_data[idx]
            data_Y = Y_data[idx]
            #pred = data_Y
            #target = 
            #x, y = Variable(data_line[:-1]), Variable(data_line[1:])
            x, y = Variable(data_line), Variable(data_Y)
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x.unsqueeze(0)).squeeze(0).transpose(-2,1)
            
            output = output.double()
            y = y.double()
            
            loss = criterion(output,y) #.double()
            #loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
             #                 torch.matmul((1-y), torch.log(1-output).float().t()))
            #loss = 
            total_loss += loss.item()
            count += output.size(0)
            
            pred = output
            target = y
            sum_all += displacement_error(pred,target)
            #N = pred.shape[0]
            #T = pred.shape[1]
            #sum_ = 0 
            #for i in range(N):
            #    for t in range(T):
                    #sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
             #       sum_+=math.sqrt((pred[i,t] - target[i,t])**2+(pred[i,t] - target[i,t])**2)
            
            #sum_all += sum_/(N*T)
            
            
        eval_loss = total_loss / count
        ade = sum_all/count
        #print("X.shape")
        #print(x.shape)
        #print(output, output.shape)
        #print("now y")
        #print(y, y.shape)
        print(name + " loss: {:.5f}".format(eval_loss))
        print(name + " ade loss: {:.5f}".format(ade))
        return eval_loss, ade



def train(ep):
    model.train()
    total_loss = 0
    count = 0
    train_total_loss = 0
    train_count = 0
    sum_all = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    ###trying
    #training_loss = []
    
    for idx in train_idx_list:
        data_line = X_train[idx]
        data_Y = Y_train[idx]
        x, y = Variable(data_line), Variable(data_Y)
        #x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x.unsqueeze(0)).squeeze(0).transpose(-2,1)
        
        output = output.double()
        y = y.double()
        
        loss = criterion(output,y) #.double()
        #loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
         #                   torch.matmul((1 - y), torch.log(1 - output).float().t()))
        total_loss += loss.item()
        count += output.size(0)
        
        train_total_loss += loss.item()
        train_count += output.size(0)
        
        pred = output
        target = y
        sum_all += displacement_error(pred,target)
        #N = pred.shape[0]
        #T = pred.shape[1]
        #sum_ = 0 
        #for i in range(N):
        #    for t in range(T):
         #       #sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
         #       sum_+=math.sqrt((pred[i,t] - target[i,t])**2+(pred[i,t] - target[i,t])**2)
        
        #sum_all += sum_/(N*T)
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = total_loss / count
            print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
            total_loss = 0.0
            count = 0
            #training_loss.append(cur_loss)
    training_loss = train_total_loss/train_count
    tade = sum_all/train_count
    print("training loss: {:.5f}".format(training_loss))
    print("training ade: {:.5f}".format(tade))
    return training_loss, tade

def plot_curves(training_losses, val_losses):
    plt.figure()
    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig("loss.svg", format="svg")
    
def plot_curves_ade(tade, vade):
    plt.figure()
    plt.plot(range(len(tade)), tade)
    plt.plot(range(len(vade)), vade)
    plt.ylabel("ADE")
    plt.xlabel("Epoch")
    plt.savefig("ADE.svg", format="svg")

if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    trainingloss_list =[]
    tade_list = []
    vade_list = []
    model_name = "poly_music_{0}.pt".format(args.data)
    for ep in range(1, args.epochs+1):
        trainingloss, tade = train(ep)
        r = tade.cpu().detach().numpy()
        vloss,vade = evaluate(X_valid,Y_valid, name='Validation')
        s = vade.cpu().detach().numpy()
        tloss = evaluate(X_test, Y_test, name='Test')
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model, f)
                print("Saved model!\n")
            best_vloss = vloss
        if ep > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        vloss_list.append(vloss)
        trainingloss_list.append(trainingloss)
        vade_list.append(s)
        tade_list.append(r)

    print('-' * 89)
    model = torch.load(open(model_name, "rb"))
    tloss = evaluate(X_test, Y_test, name='Test')
    plot_curves(trainingloss_list,vloss_list)
    plot_curves_ade(tade_list,vade_list)

