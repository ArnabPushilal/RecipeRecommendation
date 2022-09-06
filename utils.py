
import pandas as pd
import json
import re
import ast
import numpy as np
from collections import defaultdict
import time
from extracting import extract_ing
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix as cm
import pickle

#import pickle5 as pickle

def strar(df,column_name):

    """
    Changes string to array
    
    """

    try:
        df[column_name]= df[column_name].apply(lambda x: ast.literal_eval(x))
    except:

        df[column_name]=df[column_name].apply(lambda x: json.loads(x))

    return df


def label_fat(row):

    if row['fat']<=3:
        return 0
    elif row['fat']>3 and row['fat']<=17.5:
        return 1
    elif row['fat']>17.5:
        return 2
def label_sodium(row):

    if row['sodium']<=0.3:
        return 0
    elif row['sodium']>0.3 and row['sodium']<=1.5:
        return 1
    elif row['sodium']>1.5:
        return 2

def label_sugar(row):

    if row['sugar']<=5:
        return 0
    elif row['sugar']>5 and row['sugar']<=22.5:
        return 1 
    elif row['sugar']>22.5:
        return 2 
def label_satfat(row):

    if row['satfat']<=1.5:
        return 0
    elif row['satfat']>1.5 and row['satfat']<=5:
        return 1
    elif row['satfat']>5:
        return 2

def label_sodium_(row):

    if row['sodium']<=0.3:
        return row['sodium']/0.3
    elif row['sodium']>0.3 and row['sodium']<=1.5:
        return 1 + row['sodium']/1.5
    elif row['sodium']>1.5 and row['sodium']<=7.4:
        return 2 + row['sodium']/7.4
        
    elif row['sodium']>7.4:
        return 3
        

def label_sugar_(row):

    if row['sugar']<=5:
        return row['sugar']/5
    elif row['sugar']>5 and row['sugar']<=22.5:
        return 1 + row['sugar']/22.5
    elif row['sugar']>22.5:
        return 2 + row['sugar']/249.5

def label_fat_(row):

    if row['fat']<=3:
        return 0
    elif row['fat']>3 and row['fat']<=17.5:
        return 1
    elif row['fat']>17.5:
        return 2

def label_satfat_(row):

    if row['satfat']<=1.5:
        return row['satfat']/1.5
    elif row['satfat']>1.5 and row['satfat']<=5:
        return 1 + row['satfat']/5
    elif row['satfat']>5:
        return 2 + row['satfat']/31.2





def pad_sentence(df,column):

    """
    Pads sentence to max length
    with token '[PAD]'

    """

    pass
def find_tot_ing(df_in):

    df=df_in.drop_duplicates(subset='recipe_id')
    tot_ing=df.explode('ingredients')
    count_ings=dict(tot_ing['ingredients'].value_counts())
  
    return len(count_ings)

def calc_adj_f1(test):
    test['sig']=test.apply(lambda x: sigmoid(x['pred_prob']),axis=1)
    thresholds = np.arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [f1_score(test['rating'], to_labels(test['sig'], t),average='macro') for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)
    print("Best Thresh: ",thresholds[ix])
    test["npred"] = np.where(test.sig <= thresholds[ix], 0,1)
    print(cm(test['rating'],test['npred']))
    print("Best F1 score with readjustment of threshold",f1_score(test['rating'],test['npred']))
   

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

def pad_ingredients(df):

    """
    Helper function to pad ingredients
    
    """
    #df=strar(df,'ingredients')
    max=int(np.max(df['n_ingredients']))
    df['ingredients']=df['ingredients'].apply(lambda x: x + ['pad'] * (max - len(x)) )
    
    return df

def pad_tok_tags(df):

    df=strar(df,'tags')
    with open('tag_2_idx.pickle', 'rb') as handle:
     tag_2_idx = pickle.load(handle)

    df['tags_tok']=df['tags'].apply(lambda x:[tag_2_idx[x[i]] for i in range(len(x))])
    df['tags_tok']=df['tags_tok'].apply(lambda x: x + [0] * (68 - len(x)) )
    return df
    

def add_ingredients(df):

    """
    Function that adds indgredients from the a list
    of ingredients online
    
    """
    df=strar(df,'ingredients')
    tot_ing=df.explode('ingredients')
    count_ings=dict(tot_ing['ingredients'].value_counts())
    rare_ing=[word for word, occurrences in count_ings.items() if occurrences == 1]
    rare_ing_to_sub=defaultdict(list)
    ings=extract_ing()
    ings.remove("sugar")
    ings.remove("fat")
    ings.append("fat-free")
    ings.append("sugar-free")
    print("Searching for ingredients in the list")
    for line in rare_ing:
        for word in ings:
            r=re.compile(r'\b({})\b'.format(word))
            if r.search(line):
                rare_ing_to_sub[line].append(word)
    print("Adding ingredients to df")
    for idx,row in enumerate(df['ingredients']):
     for word in row:
        if word in rare_ing_to_sub:
            for ing in rare_ing_to_sub[word]:
                if ing not in rare_ing_to_sub:
                 df['ingredients'].iloc[idx].append(ing)
    return df

def filter_data(df_rec_raw,df_in,calorie_limit,time_limit,N_recipes_P_user,N_unique_P_user):

 """
 Filter data based on conditions. Also pads ingredients

 @params:
 df_recipes_raw(pandas dataframe): Raw recipes
 df_inter(pandas dataframe): Interactions df
 calorie_limit(int): Limit of calories per recipe
 time_limit(int): Limit of cooking time per recipe
 N_recipes_P_user(int): Minimum number of ratings per user
 N_unique_P_user(int): Minimum number of unique ratings per user
 
 Returns
 df: Filtered dataframe of interactions
 df_recipes_raw_filtered: Filtered dataframe of recipes
 """ 

 if not isinstance(df_rec_raw['nutrition'][0],list):
  df_rec_raw=strar(df_rec_raw,'nutrition')
 df_rec_raw['calorie']=df_rec_raw['nutrition'].apply(lambda x:x[0])
 df_recipes_raw_filtered=df_rec_raw[df_rec_raw['calorie']<calorie_limit]
 df_recipes_raw_filtered=df_recipes_raw_filtered[df_recipes_raw_filtered['minutes']<(time_limit)]

 print("---------------------------Adding Ingredient----------------------")
 df_recipes_raw_filtered=add_ingredients(df_recipes_raw_filtered)
 print("--------------------------Padding Ingredient----------------------")
 df_recipes_raw_filtered=pad_ingredients(df_recipes_raw_filtered)
 dfs=df_recipes_raw_filtered.merge(df_in,left_on='id',right_on='recipe_id')
 df=pd.DataFrame(dfs['user_id'].value_counts())
 df=df[df['user_id']>N_recipes_P_user]
 df.reset_index(inplace=True)
 dfs=dfs[dfs['user_id'].isin(df['index'])]
 dfs=dfs.groupby(['user_id']).filter(lambda x: x['rating'].nunique()>N_unique_P_user)
 dfs['description'].fillna("None",inplace=True)
 dfs['review'].fillna("None",inplace=True)
 dfs=strar(dfs,'tags')
 dfs=strar(dfs,'steps')
 dfs,ing_2_idx=map_user_to_id(dfs)

 return dfs,ing_2_idx

def map_user_to_id(df):

    recipe_2_idx={}
    idx_2_recipe={}
    user_2_idx={}
    idx_2_user={}
    i=0
    for idx,recipe_id in enumerate(df['recipe_id']):
            if recipe_id in recipe_2_idx:
                continue
            else:
                recipe_2_idx[recipe_id]=i
                idx_2_recipe[i]=recipe_id
                i+=1
    j=0
    for idx,user_id in enumerate(df['user_id']):
            if user_id in user_2_idx:
                continue
            else:
                user_2_idx[user_id]=j
                idx_2_user[j]=user_id
                j+=1
    
    df_r=df.drop_duplicates(subset='recipe_id')
    tot_ing=df_r.explode('ingredients')
    count_ings=dict(tot_ing['ingredients'].value_counts())
    ing_2_idx={}
    idx_2_ing={}
    i=0
    for idx,element in enumerate(count_ings):
        if element in ing_2_idx:
            continue
        else:
            ing_2_idx[element]=i
            idx_2_ing[i]=element
            i+=1


    df['i']=df['recipe_id'].apply(lambda x:recipe_2_idx[x])
    df['u']=df['user_id'].apply(lambda x:user_2_idx[x])
    df['ingredients_tok']=df['ingredients'].apply(lambda x:[ing_2_idx[x[i]] for i in range(len(x))])

    return df,ing_2_idx

def new_split(df):

    
    test_val=df.groupby('user_id').sample(frac=0.25)
    train=pd.concat([df, test_val]).drop_duplicates(keep=False)
    test=test_val.groupby('user_id').sample(frac=0.5)
    val=pd.concat([test_val, test]).drop_duplicates(keep=False)

    return train,val,test


def split_train_test_val(df):

    train, validate, test = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.6*len(df)), int(.8*len(df))])

    return train,validate,test

def calc_health_values(df):

    """
    Calculate health score based on provided info
    @params:
    df(pandas dataframe): Dataframe containing health info (The recipe one)
    
    returns:
    df(pandas dataframe): Dataframe
    """

    df['fat']=df['nutrition'].apply(lambda x:x[1])
    df['sugar']=df['nutrition'].apply(lambda x:x[2])
    df['sodium']=df['nutrition'].apply(lambda x:x[3])
    df['protein']=df['nutrition'].apply(lambda x:x[4])
    df['satfat']=df['nutrition'].apply(lambda x:x[5])
    df['carb']=df['nutrition'].apply(lambda x:x[6])

    df['fat']=(df.loc[:,'fat']/100)*78
    df['sodium']=(df.loc[:,'sodium']/100)*2.3
    df['sugar']=(df.loc[:,'sugar']/100)*50
    df['satfat']=(df.loc[:,'satfat']/100)*20

    df['health_fat'] = df.apply (lambda row: label_fat(row), axis=1)
    df['health_satfat'] = df.apply (lambda row: label_satfat(row), axis=1)
    df['health_sugar'] = df.apply (lambda row: label_sugar(row), axis=1)
    df['health_sodium'] = df.apply (lambda row: label_sodium(row), axis=1)
    
    df['health_satfat_2'] = df.apply (lambda row: label_satfat_(row), axis=1)
    df['health_sugar_2'] = df.apply (lambda row: label_sugar_(row), axis=1)
    df['health_sodium_2'] = df.apply (lambda row: label_sodium_(row), axis=1)


    df['tot_health']=df['health_satfat']+df['health_sugar'] + df['health_sodium']
    
    df['tot_health_2']=df['health_satfat_2']+df['health_sugar_2'] + df['health_sodium_2']

    return df




def confusion_matrix(labels,predicted,num_classes):

    """
    Returns Confusion Matrix given labels & predictions from a model.
    This function will be used in the minibatch to compute the F1 score

    @params:
    labels:(numpy array) 1D vector
    predicted:(Numpy array) Predicted vectord
    num_classes(int): The number of classes in the dataset

    """

    
    confusion= np.zeros((num_classes,num_classes))

    for i in range(len(labels)):
        confusion[labels[i]][predicted[i]] += 1 #Count labels For example
                             #(if 0,0 correctly predicted index will go up)
                             #(if 0,1 incorrectly predicted index will go up)
    copy=np.copy(confusion)

    return copy


def compute_F1(labels,predicted,num_classes):
    """
    Computes F1 score for multiclass classification, calculates individual F1 score
    based on each class, then calculates a weighted average.

    @params:
    labels:(numpy array) 1D vector
    predicted:(Numpy array) Predicted vectord
    num_classes(int): The number of classes in the dataset
    """

    conf=confusion_matrix(labels,predicted,num_classes)

    TP=np.diag(conf)
    FP = np.sum(conf, axis=0) - TP
    FN = np.sum(conf, axis=1) - TP
    TN = np.sum(conf) - (FP + FN + TP)
    eps=1e-8 # for numerical stability, otherwise we get nans
    F1= (2*TP)/(2*TP +FP +FN + eps) 

    weights=Counter(labels) #Count how many classes we have for each label

    array=np.zeros(num_classes)
    for i in range(num_classes):
        array[i]=weights[i]
    

    weighted_f1=np.sum((array * F1 ))/sum(weights.values()) #Calculate F1 score as per class weights
                                                            # This is the "average" F1 score

    return weighted_f1

def resample_df(df,kind="over"):

    count_class_1, count_class_0 = df.rating.value_counts()
   

    df_class_0 = df[df['rating'] == 0]
    df_class_1 = df[df['rating'] == 1]
    if kind == "over":
     df_class_0_over = df_class_0.sample(count_class_1//2, replace=True)
     df_test_over = pd.concat([df_class_1, df_class_0_over], axis=0)
     return df_test_over
    elif kind =="under":

     df_class_1_under = df_class_1.sample(count_class_0)
     df_test_under = pd.concat([df_class_1_under, df_class_0], axis=0)
     return df_test_under


def precision_K(r,k):
    
    assert k>=1 and k<=len(r)

    return np.mean(r[:k])


def avg_precision_K(r,K):
    
  

    #If all 0s till rank K specified just return 0
    if np.sum(r[:K])==0:
        return 0
    else:
     
      
  
     total_relevant_docs=np.sum(r)
     #Only taking K relevant ranks for consideration
     r=np.array(r)[:K]
     
     #Computing precision @ K for retrieved ranks till K
     array=np.array([precision_K(r,k+1) for k in range(len(r))])

     #Getting the array with only relevant docs
     array=array[r!=0]
     
     #min of relevant docs and K chosen 
     total_relevant_docs=min(total_relevant_docs,K)


     return np.sum(array)/total_relevant_docs


def mean_avg_precision_K(df,K):



    qids=np.unique(df['u'])
  
    precision=0

    for q in qids:

        temp_df=df[df['u']==q]
        #temp_df['relevance'].loc[temp_df['relevance'] >0 ] = 1


        precision+=avg_precision_K(temp_df['rating'],K)

        #print(precision,"prec")


    mean_avg=precision/len(qids)

    return mean_avg

def NDCG_K(df,K):

    qids=np.unique(df['u'])
    pd.options.mode.chained_assignment = None 
    NDCG_array=[]


    for q in qids:

        temp_df=df[df['u']==q]
        if np.sum(temp_df['rating'])==0:
          NDCG_array.append(0)
        else:
         temp_df['gain']=2 **(temp_df['rating']) - 1
         temp_df['gain']=temp_df['gain']
         temp_df['IDCG'] = -np.sort(-temp_df['gain'])/(np.log2(np.arange(len(temp_df))+2))
         temp_df['discounted_gain']=temp_df['gain'] / (np.log2(np.arange(len(temp_df))+2))

         temp_df=temp_df[:K]

         NDCG_array.append( np.sum(temp_df['discounted_gain'])/np.sum(temp_df['IDCG']))
    return np.mean(NDCG_array)