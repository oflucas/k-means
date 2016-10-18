
# coding: utf-8

# In[8]:

import numpy as np
from KMeans_weighted import KMeans_weighted


# In[9]:

clst = KMeans_weighted(k_clusters=2, n_init=4, debug_=True)


# In[10]:

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
W = np.array([ 1,     100,      1,      1,      1,      100])

# In[11]:

clst.fit(X, W)


# In[ ]:
# print "clustering finish, output results:"
# print "final inertia =", clst.inertia_
# print clst.print_clusters(clst.labels_)
# print clst.print_centers(clst.centers_)


