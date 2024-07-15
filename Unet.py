def unet_model(input_size):
    inputs = Input(input_size)

    # Encoder
    down_conv1 = Conv2D(256, 3, activation=LeakyReLU(), padding='same')(inputs)
    down_pool1 = AveragePooling2D(pool_size=(2, 2))(down_conv1)

    down_conv2 = Conv2D(512, 3, activation=LeakyReLU(), padding='same')(down_pool1)
    down_pool2 = AveragePooling2D(pool_size=(2, 2))(down_conv2)

    down_conv3 = Conv2D(1024, 3, activation=LeakyReLU(), padding='same')(down_pool2)
    down_pool3 = AveragePooling2D(pool_size=(2, 2),)(down_conv3)

    # Decoder
    up_conv1 = Conv2DTranspose(512, 3, activation=LeakyReLU(), padding='same')(down_pool3)
    up1 = UpSampling2D((2, 2), interpolation="mitchellcubic")(up_conv1)
    concat1 = concatenate([down_conv3, up1], axis=-1)

    up_conv2 = Conv2DTranspose(256, 3, activation=LeakyReLU(), padding='same')(concat1)
    up2 = UpSampling2D((2, 2), interpolation="mitchellcubic")(up_conv2)
    concat2 = concatenate([down_conv2, up2], axis=-1)

    up_conv3 = Conv2DTranspose(128, 3, activation=LeakyReLU(), padding='same')(concat2)
    up3 = UpSampling2D((2, 2), interpolation="mitchellcubic")(up_conv3)
    concat3 = concatenate([down_conv1, up3], axis=-1)

    outputs = Conv2D(7, 1, activation=LeakyReLU())(concat3)
    outputs0 = Dense(128, activation=LeakyReLU())(outputs)
    outputs1 = Dense(32, activation=LeakyReLU())(outputs0)
    outputs2 = Dense(7, activation='linear')(outputs1)


    model = Model(inputs=inputs, outputs=outputs2)
    return model

