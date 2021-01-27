import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

daneTreningowe = pd.read_csv("fashion_mnist_train_images.csv")
daneTestowe = pd.read_csv("fashion_mnist_test_images.csv")

etykietyTrening = pd.read_csv("fashion_mnist_train_label.csv")
etykietyTest = pd.read_csv("fashion_mnist_test_label.csv")

# daneTreningowe = daneTreningowe/255
# daneTestowe = daneTestowe/255
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
daneTreningowe = pd.DataFrame(minmax_scale.fit_transform(daneTreningowe))
daneTestowe = pd.DataFrame(minmax_scale.fit_transform(daneTestowe))

np.random.seed(0)
indeksTest = np.random.choice(60000,10000,replace = False)+
indeksTrening = list(set(range(60000)) - set(indeksTest))

x_train = daneTreningowe.iloc[indeksTrening]
y_train = etykietyTrening.iloc[indeksTrening]
y_train = np.ravel(y_train)

x_valid = daneTreningowe.iloc[indeksTest]
y_valid = etykietyTrening.iloc[indeksTest]
y_valid = np.ravel(y_valid)

# Fitting Logistic Regression
modelRL = LogisticRegression(C=10, multi_class='ovr', penalty='l2', max_iter=3000).fit(x_train, y_train)

# Training Accuracy
confusion_matrix(modelRL.predict(x_train), y_train)
modelRL_trening_acc = (modelRL.predict(x_train) == y_train).mean()
print("Training accuracy =", modelRL_trening_acc)

# Validation Accuracy
confusion_matrix(modelRL.predict(x_valid), y_valid)
modelRL_test_acc = (modelRL.predict(x_valid) == y_valid).mean()
print("Validation accuracy =", modelRL_test_acc)

print(f'Training accuracy = {modelRL_trening_acc} | Validation accuracy = {modelRL_test_acc}')



## dane valid 10000, dane trening 40000 | /256
# Training accuracy = 0.867175
# Validation accuracy = 0.8446
# Training accuracy = 0.867175 | Validation accuracy = 0.8446

## dane valid 10000, dane trening 50000 | /256
# Training accuracy = 0.86484
# Validation accuracy = 0.8556
# Training accuracy = 0.86484 | Validation accuracy = 0.8556

## dane valid 10000, dane trening 50000 | /255
# Training accuracy = 0.86698
# Validation accuracy = 0.8565
# Training accuracy = 0.86698 | Validation accuracy = 0.8565

## dane valid 10000, dane trening 50000 | prepocession
# Training accuracy = 0.86562
# Validation accuracy = 0.855
# Training accuracy = 0.86562 | Validation accuracy = 0.855

## dlugo, c=1   max_iter=2000
# Training accuracy = 0.87386
# Validation accuracy = 0.8515
# Training accuracy = 0.87386 | Validation accuracy = 0.8515

## jeszcze dluzej, c=100    max_iter=4000
# Training accuracy = 0.88008
# Validation accuracy = 0.8473
# Training accuracy = 0.88008 | Validation accuracy = 0.8473

## jeszcze dluzej, c=10    max_iter=3000
# Training accuracy = 0.87804
# Validation accuracy = 0.8497
# Training accuracy = 0.87804 | Validation accuracy = 0.8497


# shape(x_train) (50000, 784)
# shape(y_train) (50000, 1)
# Powinno byc :
# shape(x_train) (50000, 784)
# shape(y_train) (50000,)

