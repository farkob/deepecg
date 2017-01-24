from keras.models import Sequential
from keras.layers import Merge, Dense, Conv1D, Activation, Flatten, MaxPooling1D, Dropout
from keras.callbacks import ModelCheckpoint
from getXY import getXY

x_train, y_train, categories = getXY('data.pickle')

# two separete CNNs, for approximation and detail inputs
cnns = [Sequential(), Sequential(), Sequential()]

for idx, cnn in enumerate(cnns):
    shape = (12, 262) if idx != 2 else (12, 511)
    cnn.add(Conv1D(64, 3, border_mode='same', input_shape=shape))
    cnn.add(MaxPooling1D())
    cnn.add(Conv1D(32, 3, border_mode='same'))
    cnn.add(MaxPooling1D())
    cnn.add(Conv1D(16, 3, border_mode='same'))
    cnn.add(MaxPooling1D())
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())

model = Sequential()
model.add(Merge(cnns, mode='concat'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Dense(categories, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save('model.h5')

save = ModelCheckpoint("./weights/weights.hdf5", save_best_only=True,  save_weights_only=True)

model.fit(x_train, y_train, nb_epoch=100, batch_size=32, validation_split=0.1, callbacks=[save])