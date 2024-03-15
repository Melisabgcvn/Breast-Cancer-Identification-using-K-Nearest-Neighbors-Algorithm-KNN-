import pandas as pd 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

data = pd.read_csv("data.csv")


data.drop(['id','Unnamed: 32'],axis=1, inplace= True)
data = data.rename(columns = {'diagnosis' : 'target'})
data['target'] = [1 if i.strip() == 'M' else 0 for i in data.target]

# %%Outlier Tespiti
y = data.target
x = data.drop(["target"],axis = 1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score

# threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()


plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1],color = "blue", s = 50, label = "Outliers")
plt.scatter(x.iloc[:,0], x.iloc[:,1], color = "k", s = 3, label = "Data Points")

radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolors = "r",facecolors = "none", label = "Outlier Scores")
plt.legend()
plt.show()

# drop outliers
x = x.drop(outlier_index)
y = y.drop(outlier_index).values

#%%
test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_size, random_state = 42)

#%%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #fit işlemi train ile yapılır
X_test = scaler.transform(X_test) #testte tekrar fit YOK

X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe() # DataFrame'inin istatistiksel özeti
X_train_df["target"] = Y_train

data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")

plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
plt.show()
#%%

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_test, Y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic KNN Acc: ",acc)


#%%
def KNN_Best_Params(x_train, x_test, y_train, y_test):
    #daha önceden ayırdığımız verileri kullanıyoruz.
    k_range = list(range(1,31)) #en iyi komşu için bir sayı listesi
    weight_options = ["uniform","distance"] #uzaklık hesabı için formül seçenekleri
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    #artık param_grid parametreleri tutan bir sözlük yapısı 
    knn = KNeighborsClassifier() #knn modeli
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy")
    #Burada GridSearchCV ve Cross Validation kullanarak en iyi parametreleri bulucaz.
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    return grid

grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)

#%% 

import joblib 

best_model = grid.best_estimator_

# En iyi modeli kaydedin
joblib.dump(best_model, 'best_knn_model.pkl')
