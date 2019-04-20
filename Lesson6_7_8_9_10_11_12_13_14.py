import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.optimizers import SGD


# Function to create model, required for KerasClassifier
def create_model():
    # Create model
    model = Sequential()
    ...
    # Compile model
    model.compile(...)
    return model

# create classifier for use in scikit-learn
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10)

# Lesson 9:for example, you can create a dropout layer with the probability of 20% and add it to your model as follows:
model.add(Dropout(0.2))

# lesson 10 Stochastic Gradient Descent
sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False)
model.compile(..., optimizer=sgd)

# evaluate model using 10-fold cross validation in scikit-learn
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)

# Fit the model - lesson 8
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# list all data in history - lesson 7 
history = model.fit(..., callbacks=callbacks_list)
print(history.history.keys())