import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import io
from logreg_tools import *

wine_data = io.loadmat("data/wine.mat");


# Loading data and standardizing features
wine_data['X'], wine_data['y'] = data_shuffle(wine_data['X'], wine_data['y'])

training_data, training_labels, val_data, val_labels = validation_partition(wine_data['X'],wine_data['y'],5400);

training_data, val_data = standardize_features(training_data,val_data);

training_data = feature_append(training_data)

val_data = feature_append(val_data)


# Batch Gradient Descent
wine_logreg = LogisticRegression()
w, pred_labels, cost = wine_logreg.train(training_data,training_labels,lr=0.01,reg=1E-4,iterations=2000);

plt.plot(cost,'.-b')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

print("Training accuracy: ",accuracy(wine_logreg.calculate(training_data),training_labels))
print("Validation accuracy: ",accuracy(wine_logreg.calculate(val_data),val_labels))


# Mini-Batch Gradient Descent
wine_logreg_mb = LogisticRegression()
w, pred_labels, cost = wine_logreg_mb.train(training_data,training_labels,lr=0.01,reg=1E-4,iterations=2000, batch_size=100);

plt.plot(cost,'.-b')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

print("Training accuracy: ",accuracy(wine_logreg_mb.calculate(training_data),training_labels))
print("Validation accuracy: ",accuracy(wine_logreg_mb.calculate(val_data),val_labels

# Stochastic Gradient Descent
wine_logreg_sgd = LogisticRegression()
w, pred_labels, cost = wine_logreg_sgd.train(training_data,training_labels,lr=0.01,reg=1E-4,iterations=5000, batch_size=1);

plt.plot(cost,'.-b')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

print("Training accuracy: ",accuracy(wine_logreg_sgd.calculate(training_data),training_labels))
print("Validation accuracy: ",accuracy(wine_logreg_sgd.calculate(val_data),val_labels))