"""
CIS - Trainee - Período 2
Maurício S. Silva

"""


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os

dbpath = os.getcwd()+'\\winequality.csv'
wine_data = pd.read_csv(dbpath)

#prediction class 'wine_is_red' ou 'quality'
pclass = 'quality'  #caso se deseje, basta inserir 'quality' para
                        #prever a qualidade do vinho


#Preprocessing
#Retirada da coluna 'Unnamed: 0'
cols = list(wine_data.columns)
cols.remove('Unnamed: 0')
wine_data = wine_data[cols]


#Feature Scaling
#atributos considerados numéricos: todos, exceto 'quality' e 'wine_is_red'
numericalAttributes = [col for col in cols if col not in ['quality','wine_is_red']]

normalize = True   #MinMaxScaler
standard = False    #StandardScaler

if normalize and not standard:
    scaler = MinMaxScaler()
    scaler.fit(wine_data[numericalAttributes])
    wine_data[numericalAttributes] = scaler.transform(wine_data[numericalAttributes])
    
elif standard and not normalize:
    scaler = StandardScaler()
    scaler.fit(wine_data[numericalAttributes])
    wine_data[numericalAttributes] = scaler.fit_transform(wine_data[numericalAttributes])



#separar train_set e test_set
train_data, test_data = train_test_split(wine_data, test_size=0.2, random_state=42)

featureNames = list(wine_data.columns)
featureNames.remove(pclass)

train_set = train_data[featureNames]
train_labels = train_data[pclass]

test_set = test_data[featureNames]
test_labels = test_data[pclass]



#Implementação de um KNN
def myknn(test_set,train_set,train_labels,n_neighbors):
    
    test_set = np.array(test_set)
    train_set = np.array(train_set)
    train_labels = np.array(train_labels)
    
    
    res = []

    for instanceVector in test_set:
        #calcula distancias entre a instância de teste e as instâncias de treino
        dists = np.sqrt(np.power(instanceVector-train_set,2).sum(axis=1))
        dists_labels = np.append(dists.reshape(-1,1),train_labels.reshape(-1,1),axis=1)
        dists_labelsdf = pd.DataFrame(dists_labels,columns=['dist_euclidiana','labels'])
        dists_labelsdf = dists_labelsdf.sort_values('dist_euclidiana')
        
        
        kneighbors_df = dists_labelsdf.sort_values('dist_euclidiana').iloc[:n_neigh]
        classes = np.unique(train_labels)
        
        counts = []
        for c in classes:
            counts = np.append(counts,len(kneighbors_df.loc[kneighbors_df['labels']==c]))
        
        prob = counts / n_neigh
        res.append(list(prob))

    
    return res

n_neigh=5
resmyknn = myknn(test_set,train_set,train_labels,n_neigh)
resmyknn = np.array(resmyknn)


#scikit learn KNN implementation
neigh = KNeighborsClassifier(n_neighbors = n_neigh)
neigh.fit(train_set, train_labels)

resknnprob = neigh.predict_proba(test_set)
resknn = neigh.predict(test_set)
resknnprob == resmyknn

"""
array([[ True,  True],
       [ True,  True],
       [ True,  True],
       ...,
       [ True,  True],
       [ True,  True],
       [ True,  True]])
"""


if pclass == 'wine_is_red':
    #Confusion Matrix (exclusivamente para classificador binário)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for predict,label in zip(resknn,test_labels):
        #print(predict,label)
        
        if label == 1:
            if predict == 1:
                tp += 1
            else:
                fp += 1
        else:
            if predict == 0:
                tn += 1
            else:
                fn += 1
                
    total = tp+fp+tn+fn      
            

    cmknn = confusion_matrix(test_labels, resknn)
    
    #Performance measure
    cm = cmknn
    res = resknn
    
    #Accuracy
    acc = np.trace(cm)/np.sum(cm)
    
    #Precision e Recall

    precision = precision_score(test_labels, res)
    recall = recall_score(test_labels, res)

else: #classificação multiclasse
    cmknn = confusion_matrix(test_labels, resknn)
    cm = cmknn
    #Accuracy
    acc = np.trace(cm)/np.sum(cm)

""" 
knn: 'wine_is_red', Sem normalização

acc = 0.946923076923077
precision = 0.9285714285714286
recall = 0.8504672897196262


knn: 'wine_is_red', Com normalização (MinMaxScaler)

acc = 0.9923076923076923
precision = 0.987460815047022
recall = 0.9813084112149533

*Observa-se uma sensível melhora com a normalização


"""





#iii. Feature Importance - Aplicar ao algoritmo Random Forest e
#determinar quais features do dataset são mais importantes para
#o problema;


rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(train_set, train_labels)
for name, score in zip(featureNames, rnd_clf.feature_importances_):
    print(name, score)

featimpData = np.concatenate((np.array(featureNames).reshape(-1,1),
                              rnd_clf.feature_importances_.reshape(-1,1)),
                             axis=1)
featImportance_df = pd.DataFrame(featimpData,
                                 columns = ['feature','importance'])
featImportance_df.sort_values('importance',ascending = False)

"""
#knn
#Features mais importantes:
                 feature             importance
6   total sulfur dioxide      0.287769222626984
4              chlorides     0.2869356724330545
1       volatile acidity    0.12205266924031576
7                density    0.05500333205998983
"""

#iv. Aplicar a normalização nos dados e verificar os efeitos nos
#modelos;

#Basta setar True nas variáveis no início do código



#vi. Aplicar um dos métodos de Ensemble Learning e comparar os resultados.

#Ensemble Method: Bagging (bootstrap=True)
#KNeighbors

bag_clf1 = BaggingClassifier(
    KNeighborsClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf1.fit(train_set, train_labels)
y_pred1 = bag_clf1.predict(test_set)

confusion_matrix(test_labels, y_pred1)

#Measure
cm = confusion_matrix(test_labels, y_pred1)
#Accuracy
acc = np.trace(cm)/np.sum(cm)



#Ensemble Method: Bagging (bootstrap=True)
#DecisionTreeClassifier

bag_clf2 = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)

bag_clf2.fit(train_set, train_labels)
y_pred2 = bag_clf2.predict(test_set)

#Measure
cm = confusion_matrix(test_labels, y_pred2)
#Accuracy
acc = np.trace(cm)/np.sum(cm)


#Ensemble Method: Pasting (bootstrap=False)
#DecisionTreeClassifier

pasting_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators=500,
    max_samples=100, bootstrap=False, n_jobs=-1)

pasting_clf.fit(train_set, train_labels)
y_pred3 = pasting_clf.predict(test_set)

#Measure
cm = confusion_matrix(test_labels, y_pred3)
#Accuracy
acc = np.trace(cm)/np.sum(cm)




