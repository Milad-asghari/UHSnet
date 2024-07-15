from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, concatenate, LeakyReLU

def doubleconvunet(input_size):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation=LeakyReLU(), padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation=LeakyReLU(), padding='same')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation=LeakyReLU(), padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation=LeakyReLU(), padding='same')(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation=LeakyReLU(), padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation=LeakyReLU(), padding='same')(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)

    # Decoder
    up1 = Conv2DTranspose(128, 2, strides=(2, 2), activation=LeakyReLU(), padding='same')(pool3)
    up1 = concatenate([up1, conv3], axis=-1)
    conv4 = Conv2D(256, 3, activation=LeakyReLU(), padding='same')(up1)
    conv4 = Conv2D(256, 3, activation=LeakyReLU(), padding='same')(conv4)

    up2 = Conv2DTranspose(64, 2, strides=(2, 2), activation=LeakyReLU(), padding='same')(conv4)
    up2 = concatenate([up2, conv2], axis=-1)
    conv5 = Conv2D(128, 3, activation=LeakyReLU(), padding='same')(up2)
    conv5 = Conv2D(128, 3, activation=LeakyReLU(), padding='same')(conv5)

    up3 = Conv2DTranspose(32, 2, strides=(2, 2), activation=LeakyReLU(), padding='same')(conv5)
    up3 = concatenate([up3, conv1], axis=-1)
    conv6 = Conv2D(64, 3, activation=LeakyReLU(), padding='same')(up3)
    conv6 = Conv2D(64, 3, activation=LeakyReLU(), padding='same')(conv6)

    outputs = Conv2D(7, 1, activation=LeakyReLU())(conv6)
    outputs0 = Dense(128, activation=LeakyReLU())(outputs)
    outputs1 = Dense(32, activation=LeakyReLU())(outputs0)
    outputs2 = Dense(7, activation='linear')(outputs1)
    
    model = Model(inputs=inputs, outputs=outputs2)
    return model