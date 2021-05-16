from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam


def get_model(input_shape, num_labels, binary=False):
    # Construct model
    model = Sequential()

    model.add(Conv2D(16, kernel_size=2, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())

    optimizer = Adam(learning_rate=0.0001)
    # Compile the model
    if binary:
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    else:
        model.add(Dense(units=num_labels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    model.summary()

    return model
