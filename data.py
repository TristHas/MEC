import os
from os.path import join as pj
import torch
import pandas as pd
import numpy as np
from PIL import Image
from config import base_dir

experiments = {"reg"   : "CZ-8101 劣化試験 20180903～/",
               "mekki" : "CZ-8101 劣化試験(めっき板"}

def load_metadata(exp="reg"):
    """
        Sort the information of the given data folder.
        ______________________________________________
        Input:
            str exp: "reg" or "mekki": One of the two data folder
        ______________________________________________
        Output:
            pd.DataFrame: listing the date, angle, scale and full file path of every picture
    """
    assert exp in ["reg", "mekki"]
    root = pj(base_dir, experiments[exp])
    path = list(filter(lambda x:x.endswith(".jpg"), os.listdir(root)))
    date, deg, scale = list(zip(*map(lambda x:list(map(int,x[:-4].split("_"))), path)))
    path = list(map(lambda x:pj(root, x), path))
    return pd.DataFrame({"date":date, "deg":deg, "scale":scale, "path":path})

def get_dataset(exp="reg", deg=0, scale=1000):
    """
        Creates the training and validation dataset from the files on disk.
        __________________________________________________________________
        Input:
            str exp: 
                "reg" or "mekki"
                One of the two experiment data folder
            int deg:
                0 or 45
                The angle of the picture taken
            int scale:
                1000 or 3000
                The zooming factor of the picture
        __________________________________________________________________
        Output:
            torch.cuda.FloatTensor train:
                The training tensor
                size = 14 x 960 x 960
                The first dimension is the batch dimension.
                The first sample (index=0) is the picture of the first week. 
                The last sample (index = 13) is the picture of the last week
                The second and third dimension are the height and width of the picture.
            torch.cuda.FloatTensor val:
                The validation tensor
                size = 14 x ? x ?
                The first dimension is the batch dimension:
                The first sample (index=0) is the picture of the first week. 
                The last sample (index = 13) is the picture of the last week
                The second and third dimension are the height and width of the picture.
            torch.cuda.FloatTensor test:
                The test tensor
                size = 14 x ? x ?
                The first dimension is the batch dimension.
                The first sample (index=0) is the picture of the first week. 
                The last sample (index = 13) is the picture of the last week
                The second and third dimension are the height and width of the picture.
            torch.cuda.LongTensor tr_lbl:
                value = [0,1,2,3,...,13]
                The label corresponding to each training batch sample
            torch.cuda.LongTensor val_lbl:
                value = [0,1,2,3,...,13]
                The label corresponding to each validation batch sample
            torch.cuda.LongTensor te_lbl:
                value = [0,1,2,3,...,13]
                The label corresponding to each test batch sample
        __________________________________________________________________
        FIXME: day delta.
            The time delta between two pictures is not exactly 7 days.
            Sometimes it is 8. Sometimes it may be other.
            It might be useful to ajdust this information for regression.
    """
    assert deg in [0, 45]
    assert scale in [1000, 3000]
    # Load full dataset
    data   = load_metadata(exp)
    data   = data[data.deg==deg]
    data   = data[data.scale==scale]
    data   = data.sort_values(by="date").path.tolist()
    data   =  np.concatenate(list(map(lambda x:np.array(Image.open(x))[None, :], data)))
    
    # Create splits
    train  = data[:, :, :960]
    val    = data[:, :480, 960:]
    test   = data[:, 480:, 960:]
    val    = torch.from_numpy(val).float().unsqueeze(1).cuda()
    train  = torch.from_numpy(train).float().unsqueeze(1).cuda()
    test   = torch.from_numpy(test).float().unsqueeze(1).cuda()
    tr_lbl = val_lbl = te_lbl = torch.arange(0,14).cuda()
    return train, val, test, tr_lbl, val_lbl, te_lbl