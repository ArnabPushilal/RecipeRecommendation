import time
import pandas as pd
import numpy as np
import pickle
from utils import strar
from sentence_transformers import SentenceTransformer

# #train_1=pd.read_csv('newdata/train_3.csv')
# val_1=pd.read_csv('newdata/val_3.csv')
# test_1=pd.read_csv('newdata/test_3.csv')
model = SentenceTransformer('all-mpnet-base-v2')
#new_df=train_1.drop_duplicates(subset='i')

ing_2_idx = pickle.load( open( "ing_2_idx_short.p", "rb" ) )

ing=dict(sorted(ing_2_idx.items(), key=lambda item: item[1]))
ings=list(ing.keys())
model.to('cuda')
ing_emb=model.encode(ings)
with open('ing_emb.npy', 'wb') as f:
    np.save(f,ing_emb )
with open('ing_emb.npy', 'rb') as f:
    a = np.load(f)
    print(a.shape)

# main_df=pd.read_csv('dfs.csv')

# #df_full=main_df.drop_duplicates(subset='i')
# #df_full=df_full[['i','steps']]

# a=time.time()
# model.to('cuda')
# review_emb=model.encode(main_df['review'])
# print(review_emb.shape)
# b=time.time()
# print(b-a)

# with open('review_emb_short.npy', 'wb') as f:
#     np.save(f,review_emb )

# with open('review_emb_short.npy', 'rb') as f:
#     a = np.load(f)
#     print(a.shape)

# val_df=val_1.drop_duplicates(subset='i')
# test_df=test_1.drop_duplicates(subset='i')


# val_df=val_1[['i','steps']]
# val_df=strar(val_df,'steps')
# val_df['steps']=val_df.apply(lambda x: (" . ").join(x['steps']),axis=1)
# val_df.reset_index(inplace=True)


# test_df=test_1[['i','steps']]
# test_df=strar(test_df,'steps')
# test_df['steps']=test_df.apply(lambda x: (" . ").join(x['steps']),axis=1)
# test_df.reset_index(inplace=True)



# a=time.time()
# model.to('cuda')
# obj_val=model.encode(val_df['steps'])
# obj_test=model.encode(test_df['steps'])
# b=time.time()

# print(b-a)


# with open('test_3.npy', 'wb') as f:
#     np.save(f,obj_test )


# with open('val_3.npy', 'wb') as f:
#     np.save(f,obj_val )
# with open('test.npy', 'rb') as f:
#     a = np.load(f)
#     print(a.sa)