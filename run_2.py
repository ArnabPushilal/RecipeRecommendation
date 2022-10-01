from torch.utils.tensorboard import SummaryWriter
from test_model import test_model, test_model_CF_b, test_model_CF_h,test_model_MF,test_model_CF
import torch
import torch.nn as nn
from dataloader import urdata,neuralCFdata
import torch
import pandas as pd
from models import MF,NeuralCF_low_without_recipe_feat_combo, MF_ing,NeuralCF, NeuralCF_2,  NeuralCF_conv,NeuralCF_h, NeuralCF_h_tags, NeuralCF_low, NeuralCF_low_without_recipe, NeuralCF_new_self, NeuralCF_new_self_with_recipe_att, NeuralCF_og, NeuralCF_pre_trained
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

def run(train_path,val_path,test_path,model_ob,trainer,tester,model_name,learning_rate,epochs):
 
 log_name=model_name
 date=datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
 writer = SummaryWriter('logs/{}'.format(log_name))

 a=time.time()
 train=pd.read_csv(train_path)
 val=pd.read_csv(val_path)
 test=pd.read_csv(test_path)

#  train=train[:100]
#  val=val[:100]
#  test=test[:100]
 b=time.time()

 print("Time taken for read: ",b-a)


 train=strar(train,'ingredients')
 val=strar(val,'ingredients')
 test=strar(test,'ingredients')

 train=strar(train,'nutrition')
 val=strar(val,'nutrition')
 test=strar(test,'nutrition')

 train=strar(train,'ingredients_tok')
 val=strar(val,'ingredients_tok')
 test=strar(test,'ingredients_tok')

 train=calc_health_values(train)
 test=calc_health_values(test)
 val=calc_health_values(val)

#  train=pad_tok_tags(train)
#  test=pad_tok_tags(test)
#  val=pad_tok_tags(val)



 c=time.time()



 print("Time taken for pre-process: ",c-b)




 train.reset_index(inplace=True)
 val.reset_index(inplace=True)
 test.reset_index(inplace=True)

 count_1,count_0=train['rating'].value_counts()

 trainload=neuralCFdata(train) 
 valload=neuralCFdata(val)
 testload=neuralCFdata(test)



 trainloader = torch.utils.data.DataLoader(
    trainload,
    batch_size=12,
    shuffle=True)
 valloader = torch.utils.data.DataLoader(
       valload,
       batch_size=12,
       shuffle=True)
 testloader = torch.utils.data.DataLoader(
       testload,
       batch_size=12)
 
#Code for non-MF model (uncomment for non-mf models)
#  model=model_ob(3705,256,45276,8450)

# Code for MF model (Use this when training MF based models or uncomment the model=model_ob above)
 try:
    isinstance(model_ob(1,1,1,False),MF) 
    model=model_ob(3705,45276,256,bias=True)
 except:
    model=model_ob(3705,256,45276,8450,bias=True)


 optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
 criterion= nn.BCEWithLogitsLoss(weight=torch.tensor([count_0/count_1]).to('cuda'))
 best_m=trainer(trainloader,valloader,model,optimizer,criterion,writer,epochs=epochs,device='cuda')
 preds,preds_prob=tester(testloader,best_m,optimizer,criterion,'cuda')

#  torch.save(best_m.state_dict(), f'new_model_short/{model_name}.pt')

 preds_n=[]
 for ob in preds:
  preds_n+= ob.tolist()
 test['pred']=preds_n

 preds_p=[]
 for ob in preds_prob:
  preds_p+= ob.tolist()
 test['pred_prob']=preds_p

#  test.to_csv(f'result_df_short_time/{model_name}.csv')

 test['sig']=test.apply(lambda x: sigmoid(x['pred_prob']),axis=1)
 fpr, tpr, thresholds = metrics.roc_curve(test['rating'], test['sig'])
 roc_auc = metrics.auc(fpr, tpr)
 print(roc_auc,"test_roc")

if __name__=='__main__':

    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_h_tags,train_CF,test_model_CF,'self_attention_with_tags_1',0.001,10)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_h_tags,train_CF,test_model_CF,'self_attention_with_tags_2',0.001,10)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_h_tags,train_CF,test_model_CF,'self_attention_with_tags_3',0.001,10)
    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',MF_ing,train_CF,test_model_CF,'MF_with_ing_bias_true_1',0.001,1)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',MF_ing,train_CF,test_model_CF,'MF_with_ing_bias_true_2',0.001,15)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',MF_ing,train_CF,test_model_CF,'MF_with_ing_bias_true_3',0.001,15)


    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_h,train_CF,test_model_CF,'self_attention_1',0.001,1)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_h,train_CF,test_model_CF,'self_attention_2',0.001,8)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_h,train_CF,test_model_CF,'self_attention_3',0.001,8)

    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_h_batch,train_CF,test_model_CF,'self_attention_1_bn',0.001,10)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_h_batch,train_CF,test_model_CF,'self_attention_2_bn',0.001,10)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_h_batch,train_CF,test_model_CF,'self_attention_3_bn',0.001,10)

    
    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_new_self_with_recipe_att,train_CF,test_model_CF,'self_attention_1_with_recipe',0.001,1)
        
    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_new_self_with_recipe_att,train_CF,test_model_CF,'self_attention_2_with_recipe',0.001,10)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_new_self_with_recipe_att,train_CF,test_model_CF,'self_attention_3_with_recipe',0.001,10)



    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_2,train_CF_2,test_model_CF,'CF_basic_1',0.001,1)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_2,train_CF_2,test_model_CF,'CF_basic_2',0.001,8)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_2,train_CF,test_model_CF,'CF_basic_3',0.001,8)

    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_og,train_CF,test_model_CF,'CF_og_1',0.001,1)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_og,train_CF,test_model_CF,'CF_og_2',0.001,8)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_og,train_CF,test_model_CF,'CF_og_3',0.001,8)

    run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
        'new_data_short/test_1.csv',MF,train_MF,test_model_MF,'MF_bias_false_1',0.001,1)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',MF,train_MF,test_model_MF,'MF_bias_false_2',0.001,10)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',MF,train_MF,test_model_MF,'MF_bias_false_3',0.001,10)

    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_low_without_recipe,train_CF,test_model_CF,'att_paper_no_recipe_1',0.001,1)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_low_without_recipe,train_CF,test_model_CF,'att_paper_no_recipe_2',0.001,8)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_low_without_recipe,train_CF,test_model_CF,'att_paper_no_recipe_3',0.001,8)44

    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_low_without_recipe_feat_combo,train_CF,test_model_CF,'att_paper_no_recipe_feat_c_1',0.001,8)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_low_without_recipe_feat_combo,train_CF,test_model_CF,'att_paper_no_recipe_feat_c_2',0.001,8)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_low_without_recipe_feat_combo,train_CF,test_model_CF,'att_paper_no_recipe_feat_c_3',0.001,8)


    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_low,train_CF,test_model_CF,'att_paper_with_recipe_1',0.001,1+
    #     )

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_low,train_CF,test_model_CF,'att_paper_with_recipe_2',0.001,8)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_low,train_CF,test_model_CF,'att_paper_with_recipe_3',0.001,8)

    # run('new_data_short/train_1.csv','new_data_short/val_1.csv',\
    #     'new_data_short/test_1.csv',NeuralCF_pre_trained,train_CF,test_model_CF,'NeuralCF_pre_trained_1_true_Grad',0.001,8)

    # run('new_data_short/train_2.csv','new_data_short/val_2.csv',\
    #     'new_data_short/test_2.csv',NeuralCF_pre_trained,train_CF,test_model_CF,'NeuralCF_pre_trained_2_true_Grad',0.001,8)

    # run('new_data_short/train_3.csv','new_data_short/val_3.csv',\
    #     'new_data_short/test_3.csv',NeuralCF_pre_trained,train_CF,test_model_CF,'NeuralCF_pre_trained_3_true_Grad',0.001,8)
    

