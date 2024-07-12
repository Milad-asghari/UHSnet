def lr_schedule(epoch, lr):
    decay_rate = 0.9
    decay_steps = 15

    if epoch % decay_steps == 0 and epoch != 0:
        return lr * decay_rate
    else:
        return lr