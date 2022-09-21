
import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np

# BASELINE MF MODEL WITH AND WITHOUT BIAS

class MF(nn.Module):

    """
    Matrix factorization model with/without bias
    
    """

    def __init__(self,number_users,number_recipes,number_hstate,bias=False):

        """
        @params:

        number_users(int): Number of users incl train/test/val
        number_recipes(int): Number of recipes incl train/test/val
        number_hstate(int): Dimensionality of user/recipe embedding
        bias(bool): Include user and recipe bias in the model 

        """

        super().__init__()

        self.bias=bias

        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)

        if bias:
            self.user_bias = nn.Embedding(number_users+1, 1, padding_idx=0)
            self.recipe_bias = nn.Embedding(number_recipes+1, 1, padding_idx=0)

        
    def forward(self,x):

        """

        @params:
        x (Input dict): Values Containing batch of recipe ratings, id & user ids
        
        """

    
        score = (self.user_embedding(x['user']) * self.recipes_embedding(x['recipe'])).sum(1)

        if self.bias:
            score=score[:,None]
          
            score=score+self.user_bias(x['user']) +self.recipe_bias(x['recipe'])

            score=score.squeeze()
        
        
        return score
    
#CONTENT BOOSTED MF

class MF_ing(nn.Module):

    """
    Matrix factorization model with/without bias and ingredient information
    
    """

    def __init__(self,number_users,number_hstate,number_recipes,number_ing,bias=False):

        """
        @params:

        number_users(int): Number of users incl train/test/val
        number_recipes(int): Number of recipes incl train/test/val
        number_hstate(int): Dimensionality of user/recipe embedding
        bias(bool): Include user and recipe bias in the model 

        """

        super().__init__()

        self.bias=bias

        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.ing = nn.Embedding(number_ing+1, number_hstate, padding_idx=0)

        if bias:
            self.user_bias = nn.Embedding(number_users+1, 1, padding_idx=0)
            self.recipe_bias = nn.Embedding(number_recipes+1, 1, padding_idx=0)

        
    def forward(self,x):

        """

        @params:
        x (Input dict): Values Containing batch of recipe ratings, id & user ids
        
        """

        recipe_emb=torch.sum(self.ing(x['ingredients']),dim=1)
        score = (self.user_embedding(x['user']) * recipe_emb).sum(1)

        if self.bias:
            score=score[:,None]
          
            score=score+self.user_bias(x['user']) +self.recipe_bias(x['recipe'])

            score=score.squeeze()
        
        
        return score
    



#BASELINE NCF
    
class NeuralCF_og(nn.Module):

    def __init__(self,number_users,number_hstate,number_recipes,number_ingredients):
    

        super(NeuralCF_og,self).__init__()
        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)
        #self.ingredients_embedding=nn.Embedding(number_ingredients+1,number_hstate, padding_idx=0)
        
#        self.conv1=nn.Conv1d()

        self.lin1=nn.Linear(number_hstate*2,100)
        self.lin2=nn.Linear(100,50)
        self.lin3=nn.Linear(50,1)
        self.dropout_1=nn.Dropout(p=0.3)

        # self.lin21=nn.Linear(number_hstate,100)
        # self.lin22=nn.Linear(100,50)

        # self.lin31=nn.Linear(number_hstate,100)
        # self.lin32=nn.Linear(100,50)
        # self.dropout_2=nn.Dropout(p=0.3)
        # self.lin33=nn.Linear(50,1)

    def forward(self,x):
        

        #ing_emb=self.ingredients_embedding(x['ingredients'])

        recipe_emb=self.recipes_embedding(x['recipe'])
        user_emb=self.user_embedding(x['user'])

        feat_concat=torch.concat((recipe_emb,user_emb),dim=1)
      

        r1=self.lin1(feat_concat)
        r1=F.relu(r1)
        r1=self.dropout_1(r1)
        r1=self.lin2(r1)
        
        r1=F.relu(r1)
        r1=self.lin3(r1)

        
        # r2=self.lin31(user_emb)
        # r2=F.relu(r2)
        # r2=self.dropout_2(r2)
        # r2=self.lin32(r2)
        # r2=F.relu(r2)
        # r2=self.lin33(r2)



        score = r1 #.sum(1)
        #print(score.shape,"shape")

     
        return torch.squeeze(score)

#TWO TOWER MODEL

class NeuralCF_2(nn.Module):

    def __init__(self,number_users,number_hstate,number_recipes,number_ingredients):
    

    
        super(NeuralCF_2,self).__init__()
        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)
        #self.ingredients_embedding=nn.Embedding(number_ingredients+1,number_hstate, padding_idx=0)
        

#        self.conv1=nn.Conv1d()

        self.lin1=nn.Linear(number_hstate,100)
        self.lin2=nn.Linear(100,50)
        self.lin3=nn.Linear(50,1)
        self.dropout_1=nn.Dropout(p=0.3)

        # self.lin21=nn.Linear(number_hstate,100)
        # self.lin22=nn.Linear(100,50)

        self.lin31=nn.Linear(number_hstate,100)
        self.lin32=nn.Linear(100,50)
        self.dropout_2=nn.Dropout(p=0.3)
        self.lin33=nn.Linear(50,1)

    def forward(self,x):
        

        #ing_emb=self.ingredients_embedding(x['ingredients'])

        recipe_emb=self.recipes_embedding(x['recipe'])

        r1=self.lin1(recipe_emb)
        r1=F.relu(r1)
        r1=self.dropout_1(r1)
        r1=self.lin2(r1)
        
        r1=F.relu(r1)
        r1=self.lin3(r1)
      

        user_emb=self.user_embedding(x['user'])
        r2=self.lin31(user_emb)
        r2=F.relu(r2)
        r2=self.dropout_2(r2)
        r2=self.lin32(r2)
        r2=F.relu(r2)
        r2=self.lin33(r2)



        score = (r2 * r1) #.sum(1)
        #print(score.shape,"shape")

     
        return torch.squeeze(score)
    
#OUR MODEL

class NeuralCF_h(nn.Module):

    def __init__(self,number_users,number_hstate,number_recipes,number_ingredients):
    

    
        super(NeuralCF_h,self).__init__()
        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)
        self.ingredients_embedding=nn.Embedding(number_ingredients+1,number_hstate, padding_idx=0)
        self.pad_idx=0
        

#        self.conv1=nn.Conv1d()
        # trans_i = nn.TransformerEncoderLayer(d_model=number_hstate, nhead=8,batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(trans_i, num_layers=4)


        self.att_i=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0)
        self.att_u=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0)
        
        #self.attention=Attention(50)


        self.lin1=nn.Linear(number_hstate,100)
        self.lin2=nn.Linear(100,50)
        self.lin3=nn.Linear(50,1)
        self.health=nn.Linear(50,7)
        self.dropout_1=nn.Dropout(p=0.3)


        # self.final=nn.Linear(100,10)
        # self.final_2=nn.Linear(10,1)

        # self.lin21=nn.Linear(number_hstate,100)
        # self.lin22=nn.Linear(100,50)

        self.lin31=nn.Linear(number_hstate,100)
        self.lin32=nn.Linear(100,50)
        self.dropout_2=nn.Dropout(p=0.3)
        self.lin33=nn.Linear(50,1)

        # self.lin41=nn.Linear(number_hstate,100)
        # self.lin42=nn.Linear(100,50)
        # self.dropout_3=nn.Dropout(p=0.3)
        # self.lin43=nn.Linear(50,1)

    def forward(self,x):
        
        #user,recipe,ing,health=x

        padding_mask=(x['ingredients']==self.pad_idx)
    
        ing_emb=self.ingredients_embedding(x['ingredients'])
        #recipe_emb=self.recipes_embedding(x['recipe'])
        #recipe_emb=recipe_emb[:,None,:]
        #add_mask=torch.tensor([False]*x['ingredients'].shape[0],device='cuda')[:,None]
        #final_mask=torch.cat((padding_mask,add_mask),dim=1)
        #ing_emb=torch.cat((ing_emb,recipe_emb),dim=1)
        ing_emb_feat,weights=self.att_i(ing_emb,ing_emb,ing_emb,key_padding_mask=padding_mask)

        #ing_emb_feat=ing_emb+ing_emb

        #ing_emb=self.transformer_encoder(ing_emb,src_key_padding_mask=padding_mask)
        user_emb=self.user_embedding(x['user'])
        user_emb_1=user_emb[:,None,:]

        #print(user_emb.shape)
        # recipe_emb=self.recipes_embedding(x['recipe'])
        # recipe_emb=recipe_emb[:,None,:]
         
        recipe,weights=self.att_u(user_emb_1,ing_emb_feat,ing_emb_feat,key_padding_mask=padding_mask)
        recipe=torch.squeeze(recipe)
        #print(recipe.shape,"shape")

        # recipe_emb=self.recipes_embedding(x['recipe'])
        # r3=F.relu(self.lin41(recipe_emb))
        # r3=F.relu(self.dropout_3(self.lin42(r3)))
        # r3=self.lin43(r3)

  
        #recipe,weights=self.att_r(recipe_emb,ing_emb,ing_emb)
        #recipe=torch.squeeze(recipe)
        #recipe_emb=torch.sum(ing_emb,axis=1)
        #recipe_emb=torch.squeeze(torch.permute(i,(0,2,1))@weights)
        #print(recipe_emb.shape)
      
        r1=self.lin1(recipe)
        r1=F.relu(r1)
        r1=self.lin2(r1)
        r1=self.dropout_1(r1)
        r1=F.relu(r1)

        health=self.health(r1)

        r1=self.lin3(r1)
        
        r2=self.lin31(user_emb)
        r2=F.relu(r2)
        r2=self.dropout_2(r2)
        r2=self.lin32(r2)
        r2=F.relu(r2)
        r2=self.lin33(r2)

        # final=torch.concat((r1,r2),dim=1)

        # final=F.relu(self.final(final))
        # final=self.final_2(final)

        score = (r2 *r1) #.sum(1)
        #print(score.shape,"shape")
        return torch.squeeze(score),health

#AutoInt without recipe

class NeuralCF_low_without_recipe(nn.Module):

    def __init__(self,number_users,number_hstate,number_recipes,number_ingredients):
    

    
        super(NeuralCF_low_without_recipe,self).__init__()
        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)
        self.ingredients_embedding=nn.Embedding(number_ingredients+1,number_hstate, padding_idx=0)
        self.pad_idx=0
        
#        self.conv1=nn.Conv1d()
        #trans_i = nn.TransformerEncoderLayer(d_model=number_hstate, nhead=8,batch_first=True)
        #self.transformer_encoder = nn.TransformerEncoder(trans_i, num_layers=4)

        self.att_i=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.1)
        #self.att_u=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.1)        
        #self.attention=Attention(50)
        self.lin1=nn.Linear(number_hstate*44,100)
        self.lin2=nn.Linear(100,50)
        self.lin3=nn.Linear(50,1)
        #self.lin3=nn.Linear(50,1)
        #self.health=nn.Linear(50,7)
        self.dropout_1=nn.Dropout(p=0.3)


        self.final=nn.Linear(100,10)
        self.final_2=nn.Linear(10,1)

        # self.lin21=nn.Linear(number_hstate,100)
        # self.lin22=nn.Linear(100,50)

        self.lin31=nn.Linear(number_hstate,100)
        self.lin32=nn.Linear(100,50)
        self.dropout_2=nn.Dropout(p=0.3)
       # self.lin33=nn.Linear(50,1)

    def forward(self,x):
        
        #user,recipe,ing,health=x

        #padding_mask=(x['ingredients']==self.pad_idx)

        #add_mask=torch.tensor([False]*x['ingredients'].shape[0],device='cuda')[:,None]

        #final_mask=torch.cat((padding_mask,add_mask),dim=1)
    
        ing_emb=self.ingredients_embedding(x['ingredients'])
    
        #print(ing_emb.shape)
        
        #ing_emb,weights=self.att_i(ing_emb,ing_emb,ing_emb,key_padding_mask=padding_mask)
        #ing_emb=self.transformer_encoder(ing_emb,src_key_padding_mask=padding_mask)

        user_emb=self.user_embedding(x['user'])
        user_emb_1=user_emb[:,None,:]

        #recipe_emb=self.recipes_embedding(x['recipe'])
        #recipe_emb=recipe_emb[:,None,:]

        feature=torch.cat((ing_emb,user_emb_1),dim=1)
        feature_emb,weights=self.att_i(feature,feature,feature)
        feat_emb=feature+feature_emb #residual conn
        feat_emb=feat_emb.view(x['ingredients'].shape[0],-1)

        r1=F.relu(self.lin1(feat_emb))
        r1=F.relu(self.lin2(r1))
        r1=self.lin3(r1)

    

    
        return torch.squeeze(r1)#,health

    
#AutoInt without recipe ( 2 layers of self-attention)

class NeuralCF_low_without_recipe_feat_combo(nn.Module):

    def __init__(self,number_users,number_hstate,number_recipes,number_ingredients):
    

    
        super(NeuralCF_low_without_recipe_feat_combo,self).__init__()
        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)
        self.ingredients_embedding=nn.Embedding(number_ingredients+1,number_hstate, padding_idx=0)
        self.pad_idx=0
        
#        self.conv1=nn.Conv1d()
        #trans_i = nn.TransformerEncoderLayer(d_model=number_hstate, nhead=8,batch_first=True)
        #self.transformer_encoder = nn.TransformerEncoder(trans_i, num_layers=4)

        self.att_i=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.1)
        self.att_i_2=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.1)
        #self.att_u=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.1)        
        #self.attention=Attention(50)
        self.lin1=nn.Linear(number_hstate*44,100)
        self.lin2=nn.Linear(100,50)
        self.lin3=nn.Linear(50,1)
        #self.lin3=nn.Linear(50,1)
        #self.health=nn.Linear(50,7)
        self.dropout_1=nn.Dropout(p=0.3)


        self.final=nn.Linear(100,10)
        self.final_2=nn.Linear(10,1)

        # self.lin21=nn.Linear(number_hstate,100)
        # self.lin22=nn.Linear(100,50)

        self.lin31=nn.Linear(number_hstate,100)
        self.lin32=nn.Linear(100,50)
        self.dropout_2=nn.Dropout(p=0.3)
       # self.lin33=nn.Linear(50,1)

    def forward(self,x):
        
        #user,recipe,ing,health=x

        #padding_mask=(x['ingredients']==self.pad_idx)

        #add_mask=torch.tensor([False]*x['ingredients'].shape[0],device='cuda')[:,None]

        #final_mask=torch.cat((padding_mask,add_mask),dim=1)
    
        ing_emb=self.ingredients_embedding(x['ingredients'])
    
        #print(ing_emb.shape)
        
        #ing_emb,weights=self.att_i(ing_emb,ing_emb,ing_emb,key_padding_mask=padding_mask)
        #ing_emb=self.transformer_encoder(ing_emb,src_key_padding_mask=padding_mask)
        user_emb=self.user_embedding(x['user'])
        user_emb_1=user_emb[:,None,:]
        #recipe_emb=self.recipes_embedding(x['recipe'])
        #recipe_emb=recipe_emb[:,None,:]
        feature=torch.cat((ing_emb,user_emb_1),dim=1)
        feature_emb,weights=self.att_i(feature,feature,feature)
        # print(feature_emb.shape)
        feat_emb_2,weights=self.att_i_2(feature_emb,feature_emb,feature_emb)
        # print(feat_emb_2.shape)
       # feat_emb=feat_emb_2+feature_emb #residual conn
        feat_emb=feat_emb_2.reshape(x['ingredients'].shape[0] ,-1)

        r1=F.relu(self.lin1(feat_emb))
        r1=F.relu(self.lin2(r1))
        r1=self.lin3(r1)
        return torch.squeeze(r1)#,health

#(Auto Int with recipe embedding)

class NeuralCF_low(nn.Module):

    def __init__(self,number_users,number_hstate,number_recipes,number_ingredients):
    

    
        super(NeuralCF_low,self).__init__()
        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)
        self.ingredients_embedding=nn.Embedding(number_ingredients+1,number_hstate, padding_idx=0)
        self.pad_idx=0
        
#        self.conv1=nn.Conv1d()
        #trans_i = nn.TransformerEncoderLayer(d_model=number_hstate, nhead=8,batch_first=True)
        #self.transformer_encoder = nn.TransformerEncoder(trans_i, num_layers=4)

        self.att_i=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.1)
        #self.att_u=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.1)        
        #self.attention=Attention(50)
        self.lin1=nn.Linear(number_hstate*45,100)
        self.lin2=nn.Linear(100,50)
        self.lin3=nn.Linear(50,1)
        #self.lin3=nn.Linear(50,1)
        #self.health=nn.Linear(50,7)
        self.dropout_1=nn.Dropout(p=0.3)


        self.final=nn.Linear(100,10)
        self.final_2=nn.Linear(10,1)

        # self.lin21=nn.Linear(number_hstate,100)
        # self.lin22=nn.Linear(100,50)

        self.lin31=nn.Linear(number_hstate,100)
        self.lin32=nn.Linear(100,50)
        self.dropout_2=nn.Dropout(p=0.3)
       # self.lin33=nn.Linear(50,1)

    def forward(self,x):
        
        #user,recipe,ing,health=x

        #padding_mask=(x['ingredients']==self.pad_idx)

        #add_mask=torch.tensor([False]*x['ingredients'].shape[0],device='cuda')[:,None]

        #final_mask=torch.cat((padding_mask,add_mask),dim=1)
    
        ing_emb=self.ingredients_embedding(x['ingredients'])
    
        #print(ing_emb.shape)
        
        #ing_emb,weights=self.att_i(ing_emb,ing_emb,ing_emb,key_padding_mask=padding_mask)
        #ing_emb=self.transformer_encoder(ing_emb,src_key_padding_mask=padding_mask)

        user_emb=self.user_embedding(x['user'])
        user_emb_1=user_emb[:,None,:]

        recipe_emb=self.recipes_embedding(x['recipe'])
        recipe_emb=recipe_emb[:,None,:]

        feature=torch.cat((ing_emb,user_emb_1,recipe_emb),dim=1)
        feature_emb,weights=self.att_i(feature,feature,feature)
        feat_emb=feature+feature_emb #residual conn
        feat_emb=feat_emb.view(x['ingredients'].shape[0],-1)

        r1=F.relu(self.lin1(feat_emb))
        r1=F.relu(self.lin2(r1))
        r1=self.lin3(r1)

    

    
        return torch.squeeze(r1)#,health


class NeuralCF_h(nn.Module):

    def __init__(self,number_users,number_hstate,number_recipes,number_ingredients):
    

    
        super(NeuralCF_h,self).__init__()
        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)
        self.ingredients_embedding=nn.Embedding(number_ingredients+1,number_hstate, padding_idx=0)
        self.pad_idx=0
        

        self.att_i=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0)
        self.att_u=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0)
        
        #self.attention=Attention(50)


        self.lin1=nn.Linear(number_hstate,100)
        self.lin2=nn.Linear(100,50)
        self.lin3=nn.Linear(50,1)
        self.health=nn.Linear(50,7)
        self.dropout_1=nn.Dropout(p=0.3)


        # self.final=nn.Linear(100,10)
        # self.final_2=nn.Linear(10,1)

        # self.lin21=nn.Linear(number_hstate,100)
        # self.lin22=nn.Linear(100,50)

        self.lin31=nn.Linear(number_hstate,100)
        self.lin32=nn.Linear(100,50)
        self.dropout_2=nn.Dropout(p=0.3)
        self.lin33=nn.Linear(50,1)

        # self.lin41=nn.Linear(number_hstate,100)
        # self.lin42=nn.Linear(100,50)
        # self.dropout_3=nn.Dropout(p=0.3)
        # self.lin43=nn.Linear(50,1)

    def forward(self,x):
        
        #user,recipe,ing,health=x

        padding_mask=(x['ingredients']==self.pad_idx)
    
        ing_emb=self.ingredients_embedding(x['ingredients'])
        #recipe_emb=self.recipes_embedding(x['recipe'])
        #recipe_emb=recipe_emb[:,None,:]
        #add_mask=torch.tensor([False]*x['ingredients'].shape[0],device='cuda')[:,None]
        #final_mask=torch.cat((padding_mask,add_mask),dim=1)
        #ing_emb=torch.cat((ing_emb,recipe_emb),dim=1)
        ing_emb_feat,weights=self.att_i(ing_emb,ing_emb,ing_emb,key_padding_mask=padding_mask)

        #ing_emb_feat=ing_emb+ing_emb

        #ing_emb=self.transformer_encoder(ing_emb,src_key_padding_mask=padding_mask)
        user_emb=self.user_embedding(x['user'])
        user_emb_1=user_emb[:,None,:]

        #print(user_emb.shape)
        # recipe_emb=self.recipes_embedding(x['recipe'])
        # recipe_emb=recipe_emb[:,None,:]
         
        recipe,weights=self.att_u(user_emb_1,ing_emb_feat,ing_emb_feat,key_padding_mask=padding_mask)
        recipe=torch.squeeze(recipe)
        #print(recipe.shape,"shape")

        # recipe_emb=self.recipes_embedding(x['recipe'])
        # r3=F.relu(self.lin41(recipe_emb))
        # r3=F.relu(self.dropout_3(self.lin42(r3)))
        # r3=self.lin43(r3)

  
        #recipe,weights=self.att_r(recipe_emb,ing_emb,ing_emb)
        #recipe=torch.squeeze(recipe)
        #recipe_emb=torch.sum(ing_emb,axis=1)
        #recipe_emb=torch.squeeze(torch.permute(i,(0,2,1))@weights)
        #print(recipe_emb.shape)
      
        r1=self.lin1(recipe)
        r1=F.relu(r1)
        r1=self.lin2(r1)
        r1=self.dropout_1(r1)
        r1=F.relu(r1)

        #health=self.health(r1)

        r1=self.lin3(r1)
        
        r2=self.lin31(user_emb)
        r2=F.relu(r2)
        r2=self.dropout_2(r2)
        r2=self.lin32(r2)
        r2=F.relu(r2)
        r2=self.lin33(r2)

        # final=torch.concat((r1,r2),dim=1)

        # final=F.relu(self.final(final))
        # final=self.final_2(final)

        score = (r2 *r1) #.sum(1)
        #print(score.shape,"shape")
        return torch.squeeze(score)#,health
    
#Our model with recipe embeddings(ABLATION)

class NeuralCF_new_self_with_recipe_att(nn.Module):

    def __init__(self,number_users,number_hstate,number_recipes,number_ingredients):
    

    
        super(NeuralCF_new_self_with_recipe_att,self).__init__()
        self.user_embedding = nn.Embedding(number_users+1, number_hstate, padding_idx=0)
        self.recipes_embedding = nn.Embedding(number_recipes+1, number_hstate, padding_idx=0)
        self.ingredients_embedding=nn.Embedding(number_ingredients+1,number_hstate, padding_idx=0)
        self.pad_idx=0
        

#        self.conv1=nn.Conv1d()
        # trans_i = nn.TransformerEncoderLayer(d_model=number_hstate, nhead=8,batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(trans_i, num_layers=4)


        self.att_i=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.3)
        self.att_u=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.3)
        self.att_r=nn.MultiheadAttention(number_hstate,8,batch_first=True,dropout=0.3)
        
        #self.attention=Attention(50)


        self.lin1=nn.Linear(number_hstate,100)
        self.lin2=nn.Linear(100,50)
        self.lin3=nn.Linear(50,10)
        # self.health=nn.Linear(50,7)
        self.dropout_1=nn.Dropout(p=0.3)


        self.final=nn.Linear(30,10)
        self.final_2=nn.Linear(10,1)

        # self.lin21=nn.Linear(number_hstate,100)
        # self.lin22=nn.Linear(100,50)

        self.lin31=nn.Linear(number_hstate,100)
        self.lin32=nn.Linear(100,50)
        self.dropout_2=nn.Dropout(p=0.3)
        self.lin33=nn.Linear(50,10)

        self.lin41=nn.Linear(number_hstate,100)
        self.lin42=nn.Linear(100,50)
        self.dropout_3=nn.Dropout(p=0.3)
        self.lin43=nn.Linear(50,10)

    def forward(self,x):
        
        #user,recipe,ing,health=x

        padding_mask=(x['ingredients']==self.pad_idx)
    
        ing_emb=self.ingredients_embedding(x['ingredients'])
        ing_emb_feat,weights=self.att_i(ing_emb,ing_emb,ing_emb,key_padding_mask=padding_mask)

        user_emb=self.user_embedding(x['user'])
        user_emb_1=user_emb[:,None,:]
        recipe_emb=self.recipes_embedding(x['recipe'])
        recipe_emb_1=recipe_emb[:,None,:]
      
        feat_concat,weights=self.att_r(user_emb_1,recipe_emb_1,recipe_emb_1)
        feat_concat=torch.squeeze(feat_concat)
        r=self.lin41(feat_concat)
        r=F.relu(r)
        r=self.lin42(r)
        r=F.relu(r)
        r=self.dropout_3(r)
        r=self.lin43(r)

        recipe,weights=self.att_u(user_emb_1,ing_emb_feat,ing_emb_feat,key_padding_mask=padding_mask)
        recipe=torch.squeeze(recipe)

        #feat_concat=torch.concat((recipe,user_emb),dim=1)

      
        r1=self.lin1(recipe)
        r1=F.relu(r1)
        r1=self.lin2(r1)
        r1=self.dropout_1(r1)
        r1=F.relu(r1)

        # health=self.health(r1)
        r1=self.lin3(r1)

        r2=self.lin31(user_emb)
        r2=F.relu(r2)
        r2=self.dropout_2(r2)
        r2=self.lin32(r2)
        r2=F.relu(r2)
        r2=self.lin33(r2)

        final_cat=torch.concat((r,r1,r2),dim=1)
        final=F.relu(self.final(final_cat))
        final=self.final_2(final)

        
        
        #score = (r1*r2) #.sum(1)
        #print(score.shape,"shape")
        return torch.squeeze(final)


