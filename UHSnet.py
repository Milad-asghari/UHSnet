from keras.layers import Input, Dense, Conv2D,   Concatenate, LeakyReLU
from tensorflow.keras.models import Model


def UHSnet(input_size):
    inputs = Input(input_size)

    down_conv1 = Conv2D(16, 4, activation=LeakyReLU(), padding='same', dilation_rate=(1,1))(inputs)

    down_conv2 = Conv2D(32, 4, activation=LeakyReLU(), padding='same', dilation_rate=(2,2))(down_conv1)

    down_conv3 = Conv2D(64, 4, activation=LeakyReLU(), padding='same', dilation_rate=(4,4))(down_conv2)

    down_conv4 = Conv2D(128, 4, activation=LeakyReLU(), padding='same', dilation_rate=(8,8))(down_conv3)

    down_conv5 = Conv2D(256, 4, activation=LeakyReLU(), padding='same', dilation_rate=(16,16))(down_conv4)

    up_conv1   = Conv2D(128, 4, activation=LeakyReLU(), padding='same')(down_conv5)
    up_conv1   = Concatenate()([up_conv1, down_conv4])

    up_conv2   = Conv2D(64, 4, activation=LeakyReLU(), padding='same')(up_conv1)
    up_conv2   = Concatenate()([up_conv2, down_conv3])

    up_conv3   = Conv2D(32, 4, activation=LeakyReLU(), padding='same')(up_conv2)
    up_conv3   = Concatenate()([up_conv3, down_conv2])

    up_conv4   = Conv2D(16, 4, activation=LeakyReLU(), padding='same')(up_conv3)
    up_conv4   = Concatenate()([up_conv4, down_conv1])
    
    convout = Conv2D(7, 1, activation=LeakyReLU())(up_conv4)
    Dense1 = Dense(128, activation=LeakyReLU())(convout)
    Dense2 = Dense(32, activation=LeakyReLU())(Dense1)
    Dense3 = Dense(7, activation='linear')(Dense2)


    model = Model(inputs=inputs, outputs=Dense3)
    return model