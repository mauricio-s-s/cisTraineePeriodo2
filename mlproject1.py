import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



wine_data = datasets.load_wine(as_frame=True)
wn_data = wine_data.data
wn_labels = wine_data.target

test_data = wn_data.to_numpy()#.reshape(1, -1)
wn_data_np = wn_data.to_numpy()
wn_labels_np = wn_labels.to_numpy()

n_neigh=5


#separar train_set e test_set
#train_set, test_set = train_test_split(wn, test_size=0.2, random_state=88)


neigh = KNeighborsClassifier(n_neighbors = n_neigh)
neigh.fit(wn_data_np, wn_labels_np)


#print(neigh.predict_proba(wn_data.to_numpy()[161].reshape(1, -1)))
res1 = neigh.predict_proba(test_data)
print(res1)


def myknn(test_data,data,labels,n_neighbors):
    res = []
    
    for instanceVector in test_data:
        #calcula distancias entre o vetor e 
        dists = np.sqrt(np.power(instanceVector-data,2).sum(axis=1))
        dists_labels = np.append(dists.reshape(-1,1),labels.reshape(-1,1),axis=1)
        dists_labelsdf = pd.DataFrame(dists_labels,columns=['dist_euclidiana','labels'])
        dists_labelsdf.sort_values('dist_euclidiana')
        
        
        if int(dists_labelsdf.sort_values('dist_euclidiana').iloc[0]['dist_euclidiana'])==0:
            kneighbors_df = dists_labelsdf.sort_values('dist_euclidiana').iloc[1:n_neigh+1]
        else:
            kneighbors_df = dists_labelsdf.sort_values('dist_euclidiana').iloc[:n_neigh]
        
        classes = np.unique(labels)
        
        counts = []
        for c in classes:
            counts = np.append(counts,len(kneighbors_df.loc[kneighbors_df['labels']==c]))
        
        prob = counts / n_neigh
        res.append(list(prob))
        
        #print(res)
    return res


res2 = myknn(test_data,wn_data_np,wn_labels_np,n_neigh)
res2 = np.array(res2)
print(res2)
