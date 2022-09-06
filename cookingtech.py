import numpy as np
from scipy.spatial.distance import cdist
recipe=np.load('recipe_emb_short.npy')
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
sim=1 - cdist(recipe, recipe, metric='cosine')
sim=np.nan_to_num(sim)
sim=sim.squeeze()
print('pairwise dense output:\n {}\n'.format(sim))

