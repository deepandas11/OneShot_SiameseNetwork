import tensorflow as tf
import keras.backend as K


def lr_scheduler(epoch, model):
    if (epoch+1)%2 == 0:
        K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr)*0.99)

