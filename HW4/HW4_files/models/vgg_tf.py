from tensorflow.keras.layers import Activation, Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


def VGG(vgg_name="VGG11"):
    model = Sequential()

    # Conv and Pool Layers
    for i in range(len(cfg[vgg_name])):
        x = cfg[vgg_name][i]
        if x == 'M':
            model.add(MaxPooling2D((2, 2), strides=(2, 2), name=f'pool{i}'))
        elif i == 0:
            model.add(Conv2D(x, (3, 3), padding='same',
                      input_shape=(32, 32, 3), name=f'conv{i}'))
            model.add(Activation('relu'))
        else:
            model.add(Conv2D(x, (3, 3), padding='same', name=f'conv{i}'))
            model.add(Activation('relu'))

    # FC Layers
    model.add(Flatten())

    model.add(Dense(512, name='dense1'))
    model.add(Activation('relu'))

    model.add(Dense(512, name='dense2'))
    model.add(Activation('relu'))

    model.add(Dense(10, name='dense3'))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    # Main program will simply profile model for size
    model = VGG()
    print(model.summary())
