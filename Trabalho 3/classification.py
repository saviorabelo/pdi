# -*- coding: utf-8 -*-
# Import library
import numpy as np
from utils import Data
from extraction import Extraction
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# Import dataset
#X, Y, lb = Data.numbers()
#X, Y, lb = Extraction.huMoments()
#X, Y, lb = Extraction.lbp()
X, Y, lb = Extraction.glcm()

n_iter = 1
scores_list = []
for i in range(n_iter):
    
    # k-Fold Cross-Validation
    #model = KNeighborsClassifier(n_neighbors=10)
    #cv_scores = cross_val_score(model, X, Y, cv=5)
    #scores = np.mean(cv_scores)
    #scores_list.append(scores)
     
    # Holdout
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=30)
    #model = MLPClassifier(hidden_layer_sizes=30)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Confusion matrix
    #y_pred_inv = lb.inverse_transform(y_pred)
    #y_test_inv = lb.inverse_transform(y_test)
    #cm = confusion_matrix(y_pred_inv, y_test_inv)
    #print(cm)

    scores = accuracy_score(y_pred, y_test)
    scores_list.append(scores)

acc = np.mean(scores_list)
std = np.std(scores_list)

print('Iterations: {:d}'.format(n_iter))
print('Accuracy: {:.2f}'.format(acc*100))
print('Minimum: {:.2f}'.format(np.amin(scores_list)*100))
print('Maximum: {:.2f}'.format(np.amax(scores_list)*100))
print('Standard Deviation: {:.2f}'.format(std))