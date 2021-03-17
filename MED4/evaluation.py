import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def target_type(row):
    if row['Target'] == 0:
        return 'Voice'
    elif row['Target'] == 1:
        return 'Noise'

dataLibrary = [[1,0],[3,0],[2,0],[5,0],[1,0],[1,0],[1,0],[7,1],[8,1],[9,1],[11,1],[29,1],[19,1],[8,1],[6,1],
        [3,0],[1,0],[1,0],[5,0],[1,0],[4,0],[5,0],[7,1],[8,1],[9,1],[11,1],[29,1],[19,1],[8,1],[10,1],
        [5,0],[2,0],[1,0],[2,0],[4,0],[3,0],[5,0],[7,1],[8,1],[9,1],[11,1],[29,1],[19,1],[8,1],[9,1],
        [6,0],[2,0],[2,0],[7,0],[3,0],[2,0],[1,0],[7,1],[8,1],[16,1],[13,1],[29,1],[19,1],[45,1],[7,1],
        [7,0],[3,0],[2,0],[6,0],[1,0],[1,0],[4,0],[5,1],[8,1],[19,1],[11,1],[29,1],[19,1],[45,1],[5,1],
        [4,0],[4,0],[2,0],[3,0],[1,0],[1,0],[4,0],[5,1],[18,1],[9,1],[11,1],[29,1],[19,1],[8,1],[1,1]]

df = pd.DataFrame(dataLibrary)
df.columns = ['Pitch', 'Target']
df['Type'] = df.apply(target_type, axis=1)
df.head()
print(df)

X = df[df.columns[0:1]].values
Y = df[df.columns[1]].values

x_train,  x_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size=0.2)


knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(x_train, y_train)
knn_predictions = knn_clf.predict(x_test)
print(x_test)
print(knn_predictions)
print('Accuracy of the knn algorithm is {}'.format(accuracy_score(y_test,knn_predictions)))
print('Cross Validation Accuracy of the knn algorithm is {}'.format(cross_val_score(knn_clf, x_train, y_train, cv=5).mean()))

prediction = knn_clf.predict(np.array([[5]]))
print('Knn predicts [5] to be ' +str(prediction))

print(confusion_matrix(y_test, knn_predictions))
matrix = plot_confusion_matrix(knn_clf, x_test,y_test, cmap=plt.cm.Reds)
matrix.ax_.set_title('Confusion Matrix', color='white')
plt.xlabel('Predicted label', color='white')
plt.ylabel('True label', color='white')
#-------------fjollet, start
#plt.gcf().axes[0].tick_params(colors='white')
#plt.gcf().axes[1].tick_params(colors='white')
#plt.gcf().set_size_inches(10,6)
#-------------fjollet, stop
plt.show()

dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)
dt_predictions = dt_clf.predict(x_test)
print(dt_predictions)
print('Accuracy of the dt algorithm is {}'.format(accuracy_score(y_test,dt_predictions)))
print('Cross Validation Accuracy of the dt algorithm is {}'.format(cross_val_score(dt_clf, x_train, y_train, cv=5).mean()))

nb_clf = GaussianNB()
nb_clf.fit(x_train,y_train)
nb_predictions = dt_clf.predict(x_test)
print(nb_predictions)
print('Accuracy of the nb algorithm is {}'.format(accuracy_score(y_test,nb_predictions)))
print('Cross Validation Accuracy of the nb algorithm is {}'.format(cross_val_score(nb_clf, x_train, y_train, cv=5).mean()))
