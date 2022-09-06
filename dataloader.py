
from torch.utils.data import Dataset
import torch

class urdata(Dataset):

    """

    Pytorch dataset where one item 
    corresponds to user_id,recipe_id
    and the corresponding rating
    
    """

    def __init__(self,df):

        self.df=df
    def __getitem__(self,index):
        
        row=self.df.iloc[index]
        user=row['u']
        recipe=row['i']
        rating=row['rating']
        return {'user':user,'recipe':recipe,'rating':rating}
    def __len__(self):

        return len(self.df)

    
class neuralCFdata(Dataset):

    def __init__(self,df):

        self.df=df
    
    def __getitem__(self,index):

        row=self.df.iloc[index] 
        
        
        user=row['u']
        recipe=row['i']
        rating=row['rating']
        ingredients=torch.tensor(row['ingredients_tok'])
        health=row['tot_health']
        #tags=torch.tensor(row['tags_tok'])
        

        return {'user':user,'recipe':recipe,'rating':rating,'ingredients':ingredients,'health':health}#,'tags':tags}
    def __len__(self):

        return len(self.df)


    





    



