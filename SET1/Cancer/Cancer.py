from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers import Dense
import scikitplot as skplt
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report 
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from imblearn.metrics import sensitivity_specificity_support

import os
os.environ["PATH"] += os.pathsep + 'D:\Programy\Python36\Lib\site-packages\graphviz'

np.random.seed(2)

# number of wine classes
classifications = 2

# load dataset
dataset = np.loadtxt('Cancer.csv', delimiter=",")

# split dataset into sets for testing and training
X = dataset[:,0:9]
Y = dataset[:,9:10]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.66, random_state=5)

# convert output values to one-hot
y_train = keras.utils.to_categorical(y_train-1, classifications)
y_test = keras.utils.to_categorical(y_test-1, classifications)


# creating model
model = Sequential()
model.add(Dense(10, input_dim=9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(classifications, activation='softmax'))

# compile and fit model

sgd = RMSprop(lr=0.001, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=15, epochs=100, validation_data=(x_test, y_test))
#plot_model(model,to_file='model_plot.png', show_shapes=True, show_layer_names=True)





ynew = model.predict_classes(x_test, batch_size=15)
#ynew2 = model.predict(x_test,batch_size=15, verbose = 0 )

rounded_labels=np.argmax(y_test, axis=1)
print(len(x_test))

labels = [1,2]

print("Ten Matrix jest jakis dziwny")
cm = confusion_matrix(rounded_labels, ynew,labels )
print(cm)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



print("podsumowanie")
print(classification_report(rounded_labels, ynew) )



print("Ten Matrix jest OK")
skplt.metrics.plot_confusion_matrix(rounded_labels, ynew, normalize=False)
plt.show()

probas = model.predict_proba(x_test, batch_size=15)

#krzywa ROC
skplt.metrics.plot_roc_curve(rounded_labels, probas )  #(y_true, y_probas)
plt.show()

#print("Sens")
#print(sensitivity_specificity_support(rounded_labels, ynew, average='macro')  )


#history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(np.mean(history.history['accuracy']))
print(np.amax(history.history['accuracy']))
print(np.mean(history.history['val_accuracy']))
print(np.amax(history.history['val_accuracy']))




