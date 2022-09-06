import torch
import numpy as np
import torch.nn as nn
from utils import compute_F1

from sklearn.metrics import f1_score

def test_model(testloader,model,optimizer,criterion,device='cpu',print_=True):
    
  test_loss=[]
  model.to(device)

  test_accuracy=[]
  preds_prob=[]
  
  preds_array=[]
  test_f1=[]

  for i, batch_data in enumerate(testloader, 1):
        
         with torch.no_grad():

            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            rating=input_dict['rating'].float().type(torch.LongTensor).to(device)
    
            score= model(input_dict)
          

            
            preds=np.argmax(score.detach().cpu().numpy(),axis=1)
            test_accuracy.append(np.sum((rating.detach().cpu().numpy()==preds).astype(int))/len(rating))   
            loss = criterion(score,rating)
            label_np=rating.detach().cpu().numpy()  
      
            F1=compute_F1(label_np,preds,2)
            test_f1.append(F1)
      
          
          
    
            #test_f1.append(f1(torch.tensor(preds).cpu(),rating.cpu()))
            

            #m = nn.Softmax(dim=1)
            #prob_score=m(score)[:,1].numpy()

            #preds_array.append(prob_score)       
            test_loss.append(loss.item())
            
      
        
  if print_:

    print("-----------------------Testing Metrics-------------------------------------------")
    print("Loss: ",round(np.mean(test_loss),3))
    print("Acc: ",round(np.mean(test_accuracy),3))
    print("Weighted F1: ",round(np.mean(test_f1),3))

    return #np.array(preds_array,dtype='object')


def test_model_MF(testloader,model,optimizer,criterion,device='cpu',print_=True):
    
  test_loss=[]
  model.to(device)

  test_accuracy=[]
  preds_prob=[]
  
  
  preds_array=[]
  test_f1=[]

  for i, batch_data in enumerate(testloader, 1):
        
         with torch.no_grad():

            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device) 
            rating=input_dict['rating'].float().to(device)
    
            score= model(input_dict)
            preds_prob.append(score.detach().cpu().numpy())
        
            preds=np.round(torch.sigmoid(score.detach().cpu()).numpy())
        
            test_accuracy.append(np.sum((rating.detach().cpu().numpy()==preds).astype(int))/len(rating))   
            loss = criterion(score,rating)
            label_np=rating.detach().cpu().numpy()    
               
            F1=compute_F1(label_np.astype(int),preds.astype(int),2)
            test_f1.append(F1)
      
          
          
    
            #test_f1.append(f1(torch.tensor(preds).cpu(),rating.cpu()))
            

            #m = nn.Softmax(dim=1)
            #prob_score=m(score)[:,1].numpy()

            preds_array.append(preds)        
            test_loss.append(loss.item())
            
      
        
  if print_:

    print("-----------------------Testing Metrics-------------------------------------------")
    print("Loss: ",round(np.mean(test_loss),3))
    print("Acc: ",round(np.mean(test_accuracy),3))
    print("Weighted F1: ",round(np.mean(test_f1),3))

    return np.array(preds_array,dtype='object'),np.array(preds_prob,dtype='object')

def test_model_CF(testloader,model,optimizer,criterion,device='cpu',print_=True):
    
  test_loss=[]
  model.to(device)
  test_accuracy=[]
  preds_prob=[]
  preds_array=[]
  test_f1=[]

  for i, batch_data in enumerate(testloader, 1):
        
         with torch.no_grad():

            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            #input_dict['tags']=input_dict['tags'].to(device) 
            rating=input_dict['rating'].float().to(device)
            score= model(input_dict)
            preds_prob.append(score.detach().cpu().numpy())
            
            preds=np.round(torch.sigmoid(score.detach().cpu()).numpy())
        
            test_accuracy.append(np.sum((rating.detach().cpu().numpy()==preds).astype(int))/len(rating))   
            loss = criterion(score,rating)
            label_np=rating.detach().cpu().numpy()    
               
            F1=compute_F1(label_np.astype(int),preds.astype(int),2)
            test_f1.append(F1)
      
          
          
    
            #test_f1.append(f1(torch.tensor(preds).cpu(),rating.cpu()))
            

            #m = nn.Softmax(dim=1)
            #prob_score=m(score)[:,1].neumpy()

            preds_array.append(preds)       
            test_loss.append(loss.item())
            
      
        
  if print_:

    print("-----------------------Testing Metrics-------------------------------------------")
    print("Loss: ",round(np.mean(test_loss),3))
    print("Acc: ",round(np.mean(test_accuracy),3))
    print("Weighted F1: ",round(np.mean(test_f1),3))

    return np.array(preds_array,dtype='object'),np.array(preds_prob,dtype='object')

def test_model_CF_h(testloader,model,optimizer,criterion,health_cri,device='cpu',print_=True):
    
  test_loss=[]
  model.to(device)

  test_accuracy=[]
  health_acc=[]
  preds_prob=[]
  

  preds_array=[]
  test_f1=[]

  for i, batch_data in enumerate(testloader, 1):
        
         with torch.no_grad():

            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device)
            rating=input_dict['rating'].float().to(device)
            health_la=input_dict['health'].float().type(torch.LongTensor).to(device)
    
            score,health= model(input_dict)
            preds_prob.append(score.detach().cpu().numpy())
          
            
            preds=np.round(torch.sigmoid(score.detach().cpu()).numpy())
            preds_h=np.argmax(health.detach().cpu().numpy(),axis=1)
            health_acc.append(np.sum((health_la.detach().cpu().numpy()==preds_h).astype(int))/len(health_la)) 
        
            test_accuracy.append(np.sum((rating.detach().cpu().numpy()==preds).astype(int))/len(rating))   
            loss = criterion(score,rating) #+ health_cri(health,health_la)
            label_np=rating.detach().cpu().numpy()    
               
            F1=compute_F1(label_np.astype(int),preds.astype(int),2)
            test_f1.append(F1)
      
          
          
    
            #test_f1.append(f1(torch.tensor(preds).cpu(),rating.cpu()))
            

            #m = nn.Softmax(dim=1)
            #prob_score=m(score)[:,1].neumpy()

            preds_array.append(preds)       
            test_loss.append(loss.item())
            
      
        
  if print_:

    print("-----------------------Testing Metrics-------------------------------------------")
    print("Loss: ",round(np.mean(test_loss),3))
    print("Acc: ",round(np.mean(test_accuracy),3))
    print("Weighted F1: ",round(np.mean(test_f1),3))
    print("Health Acc: ",round(np.mean(health_acc),3))

    return np.array(preds_array,dtype='object'),np.array(preds_prob,dtype='object')


def test_model_CF_b(testloader,model,optimizer,criterion,device='cpu',print_=True):
    
  test_loss=[]
  model.to(device)

  test_accuracy=[]

  preds_prob=[]
  

  preds_array=[]
  test_f1=[]

  for i, batch_data in enumerate(testloader, 1):
        
         with torch.no_grad():

            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            #input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            input_dict['bert_emb']=input_dict['bert_emb'].to(device) 
            rating=input_dict['rating'].float().to(device)
    
            score= model(input_dict)
            preds_prob.append(score.detach().cpu().numpy())
            
            preds=np.round(torch.sigmoid(score.detach().cpu()).numpy())
        
            test_accuracy.append(np.sum((rating.detach().cpu().numpy()==preds).astype(int))/len(rating))   
            loss = criterion(score,rating)
            label_np=rating.detach().cpu().numpy()    
               
            F1=compute_F1(label_np.astype(int),preds.astype(int),2)
            test_f1.append(F1)
      
          
          
    
            #test_f1.append(f1(torch.tensor(preds).cpu(),rating.cpu()))
            

            #m = nn.Softmax(dim=1)
            #prob_score=m(score)[:,1].neumpy()

            preds_array.append(preds)       
            test_loss.append(loss.item())
            
      
        
  if print_:

    print("-----------------------Testing Metrics-------------------------------------------")
    print("Loss: ",round(np.mean(test_loss),3))
    print("Acc: ",round(np.mean(test_accuracy),3))
    print("Weighted F1: ",round(np.mean(test_f1),3))

    return np.array(preds_array,dtype='object'),np.array(preds_prob,dtype='object')


def return_prob(testloader,model,optimizer,criterion,device='cpu',print_=True):
    

  model.to(device)
  preds_prob=[]
 

  for i, batch_data in enumerate(testloader, 1):
        
         with torch.no_grad():

            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            #input_dict['tags']=input_dict['tags'].to(device) 
      
            score= model(input_dict)
            preds_prob.append(score.detach().cpu().numpy())
            
          

  return np.array(preds_prob,dtype='object')
