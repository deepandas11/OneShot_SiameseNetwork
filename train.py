import keras.backend as K

from utils import lr_scheduler


def train(n_epochs, lr, model, train_gen, val_gen, callbacks_list):

    # train_losses = list()
    # train_accuracies = list()

    for epoch in range(n_epochs):
        lr_scheduler(epoch, model)
        print("Current Learning Rate ----> ",K.get_value(model.optimizer.lr))
        model.fit_generator(generator=train_gen, callbacks=callbacks_list)

    print("Training ended")
