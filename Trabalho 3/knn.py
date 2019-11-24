import numpy as np
from utils import Data
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X, Y = Data.numbers()
scores_list = []
n_iter = 20

for i in range(n_iter):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    scores = metrics.accuracy_score(Y_test, y_pred)
    scores_list.append(scores)


acc = np.mean(scores_list)
std = np.std(scores_list)
print('Accuracy: {:.2f}'.format(acc*100))
print('Minimum: {:.2f}'.format(np.amin(scores_list)*100))
print('Maximum: {:.2f}'.format(np.amax(scores_list)*100))
print('Standard Deviation: {:.2f}'.format(std))