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
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=0.5e-3,
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







from functools import partial
from multiprocessing import Pool
#from multiprocessing import Pool

#from multiprocessing import Pool

#from fairmotion.data import amass_dip, bvh
#from fairmotion.core import motion as motion_class
#from fairmotion.tasks.motion_prediction import generate, metrics, utils
#from fairmotion.ops import conversions, motion as motion_ops

import utils, conversions
# def convert_fn_to_R(rep):
#     ops = [partial(unflatten_angles, rep=rep)]
#     if rep == "aa":
#         ops.append(partial(multiprocess_convert, convert_fn=conversions.A2R))
#     elif rep == "quat":
#         ops.append(partial(multiprocess_convert, convert_fn=conversions.Q2R))
#     elif rep == "rotmat":
#         ops.append(lambda x: x)
#     ops.append(np.array)
#     return ops

import conversions
import constants
import metrics
import amass_dip

import utils


def convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep):
    ops = utils.convert_fn_to_R(rep)
    seqs_T = [
        conversions.R2T(utils.apply_ops(seqs, ops))
        for seqs in [pred_seqs, src_seqs, tgt_seqs]
    ]
    return seqs_T


def calculate_metrics(pred_seqs, tgt_seqs):
    metric_frames = [6, 12, 18, 24]
    R_pred, _ = conversions.T2Rp(pred_seqs)
    R_tgt, _ = conversions.T2Rp(tgt_seqs)
    euler_error = metrics.euler_diff(
        R_pred[:, :, amass_dip.SMPL_MAJOR_JOINTS],
        R_tgt[:, :, amass_dip.SMPL_MAJOR_JOINTS],
        #R_pred[ :, amass_dip.SMPL_MAJOR_JOINTS],
        #R_tgt[:, amass_dip.SMPL_MAJOR_JOINTS],
        
    )
    euler_error = np.mean(euler_error, axis=0)
    mae = {frame: np.sum(euler_error[:frame]) for frame in metric_frames}
    return mae



# def test_model(model, dataset, rep, device, mean, std, max_len=None):
#     #pred_seqs, src_seqs, tgt_seqs = run_model(
#      #   model, dataset, max_len, device, mean, std,
#     #)
#     seqs_T = convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep)
#     # Calculate metric only when generated sequence has same shape as reference
#     # target sequence
#     if len(pred_seqs) > 0 and pred_seqs[0].shape == tgt_seqs[0].shape:
#         mae = calculate_metrics(seqs_T[0], seqs_T[2])
#     return seqs_T, mae


def test_model(pred_seqs, src_seqs, tgt_seqs, rep):
    #pred_seqs, src_seqs, tgt_seqs = run_model(
     #   model, dataset, max_len, device, mean, std,
    #)
    seqs_T = convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep)
    # Calculate metric only when generated sequence has same shape as reference
    # target sequence
    if len(pred_seqs) > 0 and pred_seqs[0].shape == tgt_seqs[0].shape:
        mae = calculate_metrics(seqs_T[0], seqs_T[2])
    return seqs_T, mae





#def evaluate(X_data, name='Eval'):
def evaluate(X_data,Y_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    sum_all = 0
    preds = []
    targs = []
    #preds2 = np.empty((len(X_data),24,72))
    preds2 = np.empty((len(X_data),24,96))
    #preds2 = np.empty((len(X_data),Y_data.shape[1],Y_data.shape[2]))
    #targs2 = np.empty((len(X_data),24,72))
    #targs2 = np.empty((len(X_data),Y_data.shape[1],Y_data.shape[2]))
    targs2 = np.empty((len(X_data),24,96))
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
            #preds.append(pred)
            #targs.append(target)
            pred, target = pred.cpu(),target.cpu()
            preds2[idx] = pred
            targs2[idx] = target
            
        eval_loss = total_loss / count
        ade = sum_all/count
        #print("X.shape")
        #print(x.shape)
        #print(output, output.shape)
        #print("now y")
        #print(y, y.shape)
        print(name + " loss: {:.5f}".format(eval_loss))
        print(name + " ade loss: {:.5f}".format(ade))
        return eval_loss, ade, preds2, targs2




#seqs_T[0].shape

def calc_mae(pred_seqs, src_seqs, tgt_seqs, seqs_T):
    if len(pred_seqs) > 0 and pred_seqs[0].shape == tgt_seqs[0].shape:
        R_pred, _ = conversions.T2Rp(pred_seqs)
        R_tgt, _ = conversions.T2Rp(tgt_seqs)
        mae = calculate_metrics(seqs_T[0], seqs_T[2])
    return mae #R_pred,R_tgt #mae

#R_pred, R_tgt= calc_mae(preds2, X_traintoy2, targs2, seqs_T)

def unnormalize(arr, mean, std):
    return arr * (std + constants.EPSILON) + mean


#evals, ade, preds, targs = evaluate(X_valid,Y_valid)

# preds.cpu()
# preds = np.array(preds)
# targs.cpu()
# targs = np.array(targs)
# preds2 = np.empty((len(preds),24,72))
# for i in range(len(preds)):
#     preds2[i] = preds[i]
    
# targs = Y_valids
# targs2 = np.empty((len(targs),24,72))
# for i in range(len(targs)):
#     targs2[i] = targs[i]
    

########
# X_traintoy = X_valid #[0:5]
# targs2 = targs

# ##X_traintoy2 = np.empty((len(X_traintoy),120,72))
# X_traintoy2 = np.empty((len(X_traintoy),120,96))
# for i in range(len(X_traintoy)):
#     X_traintoy2[i] = X_traintoy[i]

# mean= (X_traintoy2.mean() + targs2.mean() ) /2

# std = (X_traintoy2.std() + targs2.std() ) /2

# #import constants
# X_traintoy2 = X_traintoy2*(std + constants.EPSILON) + mean

# preds = preds*(std + constants.EPSILON) + mean

# targs = targs*(std + constants.EPSILON) + mean



# seqs_T = convert_to_T(preds, X_traintoy2, targs, "aa")
# mae= calc_mae(preds, X_traintoy2, targs, seqs_T)
# print(mae)

############

#X_valid.mean
#seqs_T = convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep)
#seqs_T = convert_to_T(output, x, y, "aa")


#euler_error = metrics.euler_diff(R_pred, R_tgt)

#print(" mae: {:.5f}".format(mae))

#print(f"Validation MAE: {mae}")



#def test_model(pred_seqs, src_seqs, tgt_seqs, rep):
#seq, mae = test_model(output, x, y, "aa")



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
            #print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
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
    # evals, ade, preds, targs = evaluate(X_train,Y_train)

    # preds2 = np.empty((len(preds),24,72))
    # for i in range(len(preds)):
    #     preds2[i] = preds[i]
        
    # targs2 = np.empty((len(targs),24,72))
    # for i in range(len(targs)):
    #     targs2[i] = targs[i]
        
    # X_traintoy = X_train #[0:5]

    # X_traintoy2 = np.empty((len(X_traintoy),120,72))
    # for i in range(len(X_traintoy)):
    #     X_traintoy2[i] = X_traintoy[i]

    # seqs_T = convert_to_T(preds2, X_traintoy2, targs2, "aa")
    # mae= calc_mae(preds2, X_traintoy2, targs2, seqs_T)
    # print("training mae=")
    # print(mae)
    for ep in range(1, args.epochs+1):
        #print("epoch")
        print("epoch: {:.2f}".format(ep))
        trainingloss, tade = train(ep)
        r = tade.cpu().detach().numpy()
        vloss,vade, preds,targs = evaluate(X_valid,Y_valid, name='Validation')
        
        if (ep %20) ==0:
            X_traintoy = X_valid #[0:5]
            targs2 = targs
    
            #X_traintoy2 = np.empty((len(X_traintoy),120,72))
            X_traintoy2 = np.empty((len(X_traintoy),120,96))
            for i in range(len(X_traintoy)):
                X_traintoy2[i] = X_traintoy[i]
    
            mean= (X_traintoy2.mean() + targs2.mean() ) /2
    
            std = (X_traintoy2.std() + targs2.std() ) /2
    
            #import constants
            X_traintoy2 = X_traintoy2*(std + constants.EPSILON) + mean
    
            preds = preds*(std + constants.EPSILON) + mean
    
            targs = targs*(std + constants.EPSILON) + mean
    
    
    
            seqs_T = convert_to_T(preds, X_traintoy2, targs, "aa")
            mae= calc_mae(preds, X_traintoy2, targs, seqs_T)
            print(mae)
            
        s = vade.cpu().detach().numpy()
        tloss,_,_,_ = evaluate(X_test, Y_test, name='Test')
        #if vloss < best_vloss:
        if (ep %10) ==0:
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
    tloss,_,_,_ = evaluate(X_test, Y_test, name='Test')
    plot_curves(trainingloss_list,vloss_list)
    plot_curves_ade(tade_list,vade_list)

