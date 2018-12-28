import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def train_fn(x, y, model, opt, crit):
    """
        Training function.
        Performs one step of gradient descent on the training data
    """
    model.train()
    out = model(x)
    acc = (out.max(1)[1]==y).float().mean().item()
    loss = crit(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return acc, loss.item()

def valid_fn(x, y, model):
    """
        Validation function.
        Evaluates the accuracy of the model on the validation data
    """
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    out = model(x)
    acc  = (out.max(1)[1]==y).float().mean().item()
    loss = crit(out, y).item()
    return acc, loss

def test_fn(x, y, model):
    """
        Test function.
        __________________________
        Output:
            Accuracy
            Confusion matrix
    """
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    out  = model(x)
    acc  = (out.max(1)[1]==y).float().mean().item()
    loss = crit(out, y).item()
    mat  = (F.softmax(out, 1)*100).int().cpu().detach().numpy()
    return acc, loss, mat

def epoch(tr_x, tr_y, val_x, val_y, model, opt, crit, n_iter=10):
    """
        Runs one "epoch" of training + validation.
        _________________________________________
        Inputs:
            tr_x  :  nclass x H x W
            tr_y  :  [0,1,2,...14]
            val_x :  nclass x H x W
            val_y :  [0,1,2,...14]
            model :  torch.nn.Module = model
            opt   :  torch.nn.optimizer
            crit  :  loss function
            n_iter:  Number of training iteration in the epoch
        _________________________________________
        Output:
            Mean training accuracy over iterations
            Mean training loss over iterations
            Validation accuracy after training
    """
    losses = []
    accs   = []
    for i in range(n_iter):
        acc, loss = train_fn(tr_x, tr_y, model, opt, crit)
        accs.append(acc)
        losses.append(loss)
    val_acc, val_loss = valid_fn(val_x, val_y, model)
    return np.mean(accs), np.mean(losses), val_acc, val_loss

def experiment(tr_x, val_x,  tr_y, val_y, model,
               n_iter=10, n_epoch=100):
    """
        Runs several epochs and monitor the results
    """
    opt  = torch.optim.Adam(model.parameters())
    crit = torch.nn.CrossEntropyLoss()
    traccs, losses, valaccs, vallosses = [],[],[], []
    for i in tqdm(range(n_epoch)):
        tracc, loss, valacc, valloss = epoch(tr_x, tr_y, val_x, val_y, model, opt, crit, n_iter)
        traccs.append(tracc)
        losses.append(loss)
        valaccs.append(valacc)
        vallosses.append(valloss)
    return traccs, losses, valaccs, vallosses