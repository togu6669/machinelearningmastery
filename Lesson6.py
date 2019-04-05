import numpy
from keras.models import Sequential
from keras.layers import Dense

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
# evaluate model using 10-fold cross validation in scikit-learn
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)