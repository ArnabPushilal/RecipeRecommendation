
import numpy as np
import time
from test_model import test_model_CF_h
import torch

def train_MF(trainloader,valloader,model,optimizer,criterion,writer,epochs=10,device='cpu',verbose=True):

     """
     Function to train the matrix factorization model given a training set,
     a validation set, optimizer & a loss function.
     Chooses best model based on validation loss.

     Input:
     trainloader(Pytorch dataloader)
     valloader(Pytorch dataloader)
     model(Pytorch model)
     optimizer(Pytorch based optimizer)
     criterion(loss function)
     epochs(int): Number of epochs
     device: 'cpu' or 'cuda'
     verbose(bool): If true prints metrics

     returns:
     best_model: Best model according to validation criterion
     """
         
     model=model.to(device) 
     val_best=np.inf
     best_model=None
        
     for epoch in range(epochs):

        train_loss = []
        val_loss=[]
        val_accuracy=[]
        train_RMSE=[]
        time_epoch=time.time() 
        for i, batch_data in enumerate(trainloader, 1):
     
            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device) 
            rating=input_dict['rating'].float().to(device)

            optimizer.zero_grad()
            score= model(input_dict)
            loss = criterion(score,rating)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        time_epoch_vl=time.time() 
        if verbose:
            print('----------------------------------------------------------------------------------')
            print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
            print("-----------------------Training Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(train_loss),4))
           # print("Acc: ",round(np.mean(RMSE),3))
    
    
        for i, batch_data in enumerate(valloader, 1):
        
         with torch.no_grad():
     
            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device) 
            rating=input_dict['rating'].float().to(device)
    
            optimizer.zero_grad()
            score= model(input_dict)
        
            loss = criterion(score,rating)
            val_loss.append(loss.item())
                
        if verbose:

            print("-----------------------Validtion Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(val_loss),4))
            #print("Acc: ",round(np.mean(val_accuracy),3))
        writer.add_scalar('Train-Epoch-Loss',round(np.mean(train_loss),4), epoch)
        writer.add_scalar('Val-Epoch-Loss',round(np.mean(val_loss),4), epoch)
        
       
        if np.mean(val_loss) < val_best:
            
            val_best=np.mean(val_loss)
            best_model=model
            if verbose:
                print("Model saved")
                #torch.save(best_model.state_dict(), 'MF_MatrixFac_baseline_new_data_2.pt')
            
    
     return best_model


def train_CF(trainloader,valloader,model,optimizer,criterion,writer,epochs=10,device='cpu',verbose=True):

     """
     Function to train the CF model given a training set,
     a validation set, optimizer & a loss function.
     Chooses best model based on validation loss.

     Input:
     trainloader(Pytorch dataloader)
     valloader(Pytorch dataloader)
     model(Pytorch model)
     optimizer(Pytorch based optimizer)
     criterion(loss function)
     epochs(int): Number of epochs
     writer(tensorboard writer)
     device: 'cpu' or 'cuda'
     verbose(bool): If true prints metrics

     returns:
     best_model: Best model according to validation criterion
     """
         
     model=model.to(device) 
     val_best=np.inf
     best_model=None
        
     for epoch in range(epochs):


        train_loss = []
        val_loss=[]
        val_accuracy=[]
        train_RMSE=[]
        time_epoch=time.time() 
        for i, batch_data in enumerate(trainloader, 1):
            # print(i,"batch")
     
            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            #input_dict['tags']=input_dict['tags'].to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device)
            #rating=input_dict['rating'].float().type(torch.LongTensor).to(device)
            rating=input_dict['rating'].float().to(device)

            optimizer.zero_grad()
            score= model(input_dict)
            loss = criterion(score,rating)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        time_epoch_vl=time.time() 
        if verbose:
            print('----------------------------------------------------------------------------------')
            print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
            print("-----------------------Training Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(train_loss),4))
           # print("Acc: ",round(np.mean(RMSE),3))
    
    
        for i, batch_data in enumerate(valloader, 1):
        
         with torch.no_grad():
     
            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            #input_dict['tags']=input_dict['tags'].to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device)
            #rating=input_dict['rating'].float().type(torch.LongTensor).to(device)
            rating=input_dict['rating'].float().to(device)
    
            optimizer.zero_grad()
            score= model(input_dict)
        
            loss = criterion(score,rating)
            val_loss.append(loss.item())
                
        if verbose:

            print("-----------------------Validtion Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(val_loss),4))
            #print("Acc: ",round(np.mean(val_accuracy),3))
        
        writer.add_scalar('Train-Epoch-Loss',round(np.mean(train_loss),4), epoch)
        writer.add_scalar('Val-Epoch-Loss',round(np.mean(val_loss),4), epoch)
      
        ### Saves models based on validation loss 
        if np.mean(val_loss) < val_best:
            
            val_best=np.mean(val_loss)
            best_model=model
            if verbose:
                print("Model saved")
                #torch.save(best_model.state_dict(), 'newmodels/NeurcalCF_og_2.pt')
            
    
     return best_model

def train_CF_2(trainloader,valloader,model,optimizer,criterion,writer,epochs=10,device='cpu',verbose=True):

     """
     Function to train the CF model given a training set,
     a validation set, optimizer & a loss function.
     Chooses best model based on validation loss.

     Input:
     trainloader(Pytorch dataloader)
     valloader(Pytorch dataloader)
     model(Pytorch model)
     optimizer(Pytorch based optimizer)
     criterion(loss function)
     epochs(int): Number of epochs
     device: 'cpu' or 'cuda'
     verbose(bool): If true prints metrics

     returns:
     best_model: Best model according to validation criterion
     """
         
     model=model.to(device) 
     val_best=np.inf
     best_model=None
        
     for epoch in range(epochs):
        

        train_loss = []
        val_loss=[]
        val_accuracy=[]
        train_RMSE=[]
        time_epoch=time.time() 
        for i, batch_data in enumerate(trainloader, 1):
            #print(i,"batch")
     
            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device)
            #rating=input_dict['rating'].float().type(torch.LongTensor).to(device)
            rating=input_dict['rating'].float().to(device)
            

            optimizer.zero_grad()
            score= model(input_dict)
            loss = criterion(score,rating)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        time_epoch_vl=time.time() 
        if verbose:
            print('----------------------------------------------------------------------------------')
            print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
            print("-----------------------Training Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(train_loss),4))
           # print("Acc: ",round(np.mean(RMSE),3))
    
    
        for i, batch_data in enumerate(valloader, 1):
        
         with torch.no_grad():
     
            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device)

            #rating=input_dict['rating'].float().type(torch.LongTensor).to(device)
            rating=input_dict['rating'].float().to(device)
    
            optimizer.zero_grad()
            score= model(input_dict)
        
            loss = criterion(score,rating)
            val_loss.append(loss.item())
                
        if verbose:

            print("-----------------------Validtion Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(val_loss),4))
            #print("Acc: ",round(np.mean(val_accuracy),3))
            
        writer.add_scalar('Train-Epoch-Loss',round(np.mean(train_loss),4), epoch)
        writer.add_scalar('Val-Epoch-Loss',round(np.mean(val_loss),4), epoch)
        
        ### Saves models based on validation loss and epochs 
        if np.mean(val_loss) < val_best:
            
            val_best=np.mean(val_loss)
            best_model=model
            if verbose:
                print("Model saved")
                #torch.save(best_model.state_dict(), 'newmodels/NeurcalCF_basic_2.pt')
            
    
     return best_model


def train_CF_health(trainloader,valloader,model,optimizer,criterion,health_cri,scheduler,epochs=10,device='cpu',verbose=True):

     """
     Function to train the CF model given a training set,
     a validation set, optimizer & a loss function.
     Chooses best model based on validation loss.

     Input:
     trainloader(Pytorch dataloader)
     valloader(Pytorch dataloader)
     model(Pytorch model)
     optimizer(Pytorch based optimizer)
     criterion(loss function)
     epochs(int): Number of epochs
     device: 'cpu' or 'cuda'
     verbose(bool): If true prints metrics

     returns:
     best_model: Best model according to validation criterion
     """
         
     model=model.to(device) 
     val_best=np.inf
     best_model=None
        
     for epoch in range(epochs):

        train_loss = []
        val_loss=[]
        val_accuracy=[]
        train_RMSE=[]
        time_epoch=time.time() 
        for i, batch_data in enumerate(trainloader, 1):
            #print(i,"batch")
     
            input_dict = batch_data 

            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device)
            #rating=input_dict['rating'].float().type(torch.LongTensor).to(device)
            rating=input_dict['rating'].float().to(device)
            health_la=input_dict['health'].float().type(torch.LongTensor).to(device) 
            
            optimizer.zero_grad()
            score,health= model(input_dict)
            loss = criterion(score,rating) #+ health_cri(health,health_la)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        time_epoch_vl=time.time() 
        if verbose:
            print('----------------------------------------------------------------------------------')
            print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
            print("-----------------------Training Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(train_loss),4))
           # print("Acc: ",round(np.mean(RMSE),3))
    
    
        for i, batch_data in enumerate(valloader, 1):
        
         with torch.no_grad():
     
            input_dict = batch_data 
            input_dict['user']=input_dict['user'].float().type(torch.LongTensor).to(device) 
            input_dict['ingredients']=input_dict['ingredients'].to(device) 
            input_dict['recipe']=input_dict['recipe'].float().type(torch.LongTensor).to(device)
            #rating=input_dict['rating'].float().type(torch.LongTensor).to(device)
            rating=input_dict['rating'].float().to(device)
            health_la=input_dict['health'].float().type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            score,health= model(input_dict)
            loss = criterion(score,rating) #+ health_cri(health,health_la)
            val_loss.append(loss.item())
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    

        #test_model_CF_h(testloader,model,optimizer,criterion,health_cri,'cuda')
                
        if verbose:

            print("-----------------------Validtion Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(val_loss),4))
            #print("Acc: ",round(np.mean(val_accuracy),3))
        
        ### Saves models based on validation loss 
        if np.mean(val_loss) < val_best:
            
            val_best=np.mean(val_loss)
            best_model=model
            if verbose:
                print("Model saved")
                #torch.save(best_model.state_dict(), 'NeuralCF_self_2_with_mask.pt')
            
    
     return best_model


