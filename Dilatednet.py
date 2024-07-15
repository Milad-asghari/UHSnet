def dilated_model(input_size):
    inputs = Input(input_size)

    down_conv1 = Conv2D(16, 4, activation=LeakyReLU(), padding='same', dilation_rate=(1,1))(inputs)

    down_conv2 = Conv2D(32, 4, activation=LeakyReLU(), padding='same', dilation_rate=(2,2))(down_conv1)

    down_conv3 = Conv2D(64, 4, activation=LeakyReLU(), padding='same', dilation_rate=(4,4))(down_conv2)

    down_conv4 = Conv2D(128, 4, activation=LeakyReLU(), padding='same', dilation_rate=(8,8))(down_conv3)

    down_conv5 = Conv2D(256, 4, activation=LeakyReLU(), padding='same', dilation_rate=(16,16))(down_conv4)

    up_conv1   = Conv2D(128, 4, activation=LeakyReLU(), padding='same')(down_conv5)

    up_conv2   = Conv2D(64, 4, activation=LeakyReLU(), padding='same')(up_conv1)

    up_conv3   = Conv2D(32, 4, activation=LeakyReLU(), padding='same')(up_conv2)

    up_conv4   = Conv2D(16, 4, activation=LeakyReLU(), padding='same')(up_conv3)

    outputs = Conv2D(7, 1, activation=LeakyReLU())(up_conv4)
    outputs1 = Dense(128, activation=LeakyReLU())(outputs)
    outputs2 = Dense(32, activation=LeakyReLU())(outputs1)
    outputs3 = Dense(7, activation='linear')(outputs2)


    model = Model(inputs=inputs, outputs=outputs3)


    return model