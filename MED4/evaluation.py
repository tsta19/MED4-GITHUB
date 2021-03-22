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


#Funktion som returnerer en string an på hvilket 'target' det har.
def target_type(row):
    if row['Target'] == 1:
        return 'Voice'
    else:
        return 'Noise'


def find_errors(testY,predictions,testX):
    for i in range(len(testY)):
        if testY[i] != predictions[i]:
            print("error with [" + str(testX[i]) + "][" + str(testY[i]) + "]")


outF = open("testEvalData.txt","r")

stringList = outF.read().split('\n')
dataList = [0 for x in range(len(stringList))]
dataLibrary = [0.0 for x in range(len(stringList))]

for i in range(len(stringList)):
    dataList[i] = stringList[i].split(', ')
    try:
        map_obj = map(float, dataList[i])
        dataLibrary[i] = list(map_obj)
    except ValueError:
        print("error on line" + str(i))

dataLibraryX = [0.0 for x in range(len(dataLibrary))]
dataLibraryY = [0.0 for x in range(len(dataLibrary))]

for i in range(len(dataLibrary)):
    dataLibraryY[i] = dataLibrary[i][0]
    dataLibraryX[i] = dataLibrary[i]
    dataLibraryX[i].pop(0)

print(len(dataLibraryY))
print(len(dataLibraryX))

"""Et library der består af et antal arrays, som hver er opdelt på følgende: [x,y]
   Her kan x f.eks. være en pitch-værdi, for et enkelt lydsample.
   Her vil y være den kategori den er blevet klassificeret som. Eks. 0 eller 1 -> 'Voice' eller 'Noice'"""
"""dataLibrary = [[1,0],[3,0],[2,0],[5,0],[1,0],[1,0],[1,0],[7,1],[8,1],[9,1],[11,1],[29,1],[19,1],[8,1],[6,1],
        [3,0],[1,0],[1,0],[5,0],[1,0],[4,0],[5,0],[7,1],[8,1],[9,1],[11,1],[29,1],[19,1],[8,1],[10,1],
        [5,0],[2,0],[1,0],[2,0],[4,0],[3,0],[5,0],[7,1],[8,1],[9,1],[11,1],[29,1],[19,1],[8,1],[9,1],
        [6,0],[2,0],[2,0],[7,0],[3,0],[2,0],[1,0],[7,1],[8,1],[16,1],[13,1],[29,1],[19,1],[45,1],[7,1],
        [7,0],[3,0],[2,0],[6,0],[1,0],[1,0],[4,0],[5,1],[8,1],[19,1],[11,1],[29,1],[19,1],[45,1],[5,1],
        [4,0],[4,0],[2,0],[3,0],[1,0],[1,0],[4,0],[5,1],[18,1],[9,1],[11,1],[29,1],[19,1],[8,1],[1,1]]"""


# Laver DataFrame hvilket inddeler det i kolonner og rækker. (Se det første som bliver printet.)
#for x in range(len(dataLibraryX)):
newArr = []
counter = 0

for x in range(len(dataLibraryX)):
    for y in range(len(dataLibraryX[x])):
        pair = []
        pair.append(dataLibraryX[x][y])
        pair.append(dataLibraryY[x])
        #newArr[counter].append(dataLibraryY[x])
        newArr.append(pair)
        counter +=1



df = pd.DataFrame(newArr)
df.columns = ['Pitch', 'Target'] # Giver en overskrift til hver kolonne i dataframen.

# Laver en ny kolonne kaldet 'Type', hvor hver værdi får tildelt enten 'Voice' eller 'Noise (ud fra om 'Target' er
# enten 0 eller 1.).
df['Type'] = df.apply(target_type, axis=1)

# df.head(5) #<- returnerer de 5 første rækker.

print(df)

X = df[df.columns[0:1]].values # X indeholder alle værdier i første række.
Y = df[df.columns[1]].values # Y indeholder alle værdier i anden række.


# Splitter datassættet op så 20 % bliver testet på mens resten (80 %) bliver trænet på.
# x_train indeholder altså alle X-værdier som bliver trænet på. x_test indeholder de X-værdier som bliver testet på.
# Det samme gør sig gældende for y_train og y_test, bare med y-værdierne i stedet.
# Dette gør at vi får en ny opdeling hver gang vi starter programmet, og fjerner noget bias.
x_train,  x_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size=0.2)


knn_clf = KNeighborsClassifier(n_neighbors=3) # Laver en K-NeighborsClassifier (Metoden)
knn_clf.fit(x_train, y_train) # Putter træningssættet ind i classifieren. Træner ud fra træningssættet.
knn_predictions = knn_clf.predict(x_test) # Predicter hvad alle testresultaterne burde være ud fra træningen.

# De næste 9 linjer kode gør bare at x_test og knn_predictions bliver printet pænere i konsollen.
str_x_test = ""
for i in range(len(x_test)):
    str_x_test += str(x_test[i])
print('x_test:        ' + str_x_test)

str_knn_predictions = ""
for i in range(len(knn_predictions)):
    str_knn_predictions += "[" + str(knn_predictions[i]) + "]"
print("Predicted y's: " + str(str_knn_predictions))
# ---------------

print("")
print("--KNN--")

# Beregner en accuracy score ud fra hvor mange predictions den har fået rigtige.
print('Accuracy of the knn algorithm is {}'.format(accuracy_score(y_test,knn_predictions)))
# Cross validation beregner accuracy scoren men ud fra at den tester 5 gange (Prøv at slette .mean() og se dens output).
print('Cross Validation Accuracy of the knn algorithm is {}'.format(cross_val_score(knn_clf, X, Y, cv=5).mean()))

# Her predicter den hvad en 5'er ville svare til i y.
prediction = knn_clf.predict(np.array([[5]]))
print(f'Knn predicts [5] to be ' + str(prediction))

# Printer en matrix i konsollen.
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))
# Laver en matrix som kan blive plottet (ud fra KNeighborsClassifieren). Den plotter testdataen.
matrix = plot_confusion_matrix(knn_clf, x_test,y_test, cmap=plt.cm.Reds)
matrix.ax_.set_title('K-Nearest Neighbors', color='black')
plt.xlabel('Predicted label', color='black')
plt.ylabel('True label', color='black')

# Gør de samme ting som med knn-classifieren så bare med en ny classifier
print("")
print("--DT--")
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)
dt_predictions = dt_clf.predict(x_test)
print('Accuracy of the dt algorithm is {}'.format(accuracy_score(y_test,dt_predictions)))
print('Cross Validation Accuracy of the dt algorithm is {}'.format(cross_val_score(dt_clf, x_train, y_train, cv=5).mean()))
prediction = dt_clf.predict(np.array([[5]]))
print(f'dt predicts [5] to be ' + str(prediction))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))
matrix = plot_confusion_matrix(dt_clf, x_test,y_test, cmap=plt.cm.Reds)
matrix.ax_.set_title('Decision Tree', color='black')
plt.xlabel('Predicted label', color='black')
plt.ylabel('True label', color='black')

# Gør de samme ting som med knn-classifieren så bare med en ny classifier
print("")
print("--GNB--")
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)
nb_predictions = nb_clf.predict(x_test)
print('Accuracy of the nb algorithm is {}'.format(accuracy_score(y_test,nb_predictions)))
print('Cross Validation Accuracy of the nb algorithm is {}'.format(cross_val_score(nb_clf, x_train, y_train, cv=5).mean()))
prediction = nb_clf.predict(np.array([[5]]))
print(f'nb predicts [5] to be ' + str(prediction))
print("Confusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))
matrix = plot_confusion_matrix(nb_clf, x_test,y_test, cmap=plt.cm.Reds)
matrix.ax_.set_title('Gaussian NB', color='black')
plt.xlabel('Predicted label', color='black')
plt.ylabel('True label', color='black')

plt.show()
