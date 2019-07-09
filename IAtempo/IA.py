import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

tempo = pd.read_csv("dados.csv", sep =';')

batch_size = 1
epochs = 35

X = tempo.iloc[:,0:7]
y = np.ravel(tempo.saida)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.to_csv('teste.csv')
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

model = Sequential()
model.add(Dense(7,input_dim = 7, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Treinamento do modelo
H = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

# Avaliação do modelo no conjunto de teste
score = model.evaluate(X_test, y_test, verbose=1)
y_pred = model.predict(X_test)

j = []
for w in range(0,len(y_pred)):
    if y_pred[w] > 0.5:
      j.append(1)
    else:
        j.append(0)

print(j)
print(y_test);

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# plotando 'loss' e 'accuracy' para os datasets 'train' e 'test'
plt.figure()
#plt.plot(np.arange(0,epochs), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0,epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,epochs), H.history["val_acc"], label="val_acc")
plt.title("Acurácia")
plt.xlabel("Épocas #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

