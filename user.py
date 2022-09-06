
from test_model import return_prob
import torch
import torch.nn as nn
from dataloader import urdata,neuralCFdata
import torch
import pandas as pd
from models import MF,NeuralCF, NeuralCF_2,  NeuralCF_conv,NeuralCF_h, NeuralCF_h_tags, NeuralCF_low, NeuralCF_low_without_recipe, NeuralCF_new_self, NeuralCF_new_self_with_recipe_att, NeuralCF_og, NeuralCF_pre_trained
import os
from train import train_CF_2, train_MF,train_CF,train_CF_health
import ast
import pickle
from utils import calc_adj_f1, calc_health_values, filter_data, pad_tok_tags,split_train_test_val,find_tot_ing,resample_df,new_split
import numpy as np
from utils import strar,sigmoid,to_labels
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
import time
import sklearn
from sklearn.metrics import f1_score
import sklearn
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import f1_score

#new_df=strar(new_df,'ingredients_tok')
def calc_ratings(user,new_df,train,model):
 
 data=new_df

 testload=neuralCFdata(data) 
 testloader = torch.utils.data.DataLoader(
       testload,
       batch_size=12)
 
 optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
 criterion= nn.BCEWithLogitsLoss()#[count_0/count_1]).to('cuda'))

 preds_prob=return_prob(testloader,model,optimizer,criterion,'cuda')
 preds_p=[]

 for ob in preds_prob:
  try:
      len(ob)
      preds_p+= ob.tolist()
  except:
      preds_p+=[ob]
 data['preds_p']=preds_p
 data['sig']=data.apply(lambda x: sigmoid(x['preds_p']),axis=1)
 return data['sig']


if __name__=="__main__":
    dfs=pd.read_csv('dfs.csv')
    new_df=dfs.drop_duplicates(subset='i')
    new_df=strar(new_df,'nutrition')
    new_df=calc_health_values(new_df)
    new_df=strar(new_df,'ingredients_tok')
    a=np.load('recipe_emb_short.npy')
    new_df['emb']=a.tolist()
    train=pd.read_csv('new_data_short/train_1.csv')
    model=NeuralCF_h(3705,256,45276,8450)
    model.load_state_dict(torch.load('new_model_short/self_attention_1.pt'))
    user_rating=np.zeros((3706,45277))
    a=time.time()
    for user in range(0,3706):
     
     new_df['u']=user
     user_rating[user]=calc_ratings(user,new_df,train,model)
     rated_=dfs[dfs['u']==user][['name','i','rating']]['i'].tolist()
     user_rating[user][rated_]=-100
    with open('user_rating_matrix.npy', 'wb') as f:
     np.save(f, user_rating)
    b=time.time()
    print("Time: ",b-a)