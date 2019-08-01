from model import siamese_architecture
from dataloader import DataGenerator
from train import *

import argparse
from datetime import datetime
import keras

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=0.001, type=float, help='initial LR')
parser.add_argument('--n_epochs', default=10, type=int, help='Epochs to train')
parser.add_argument('--b', default=32, type=int, help='Batch Size')


def main(args):

    print("Starting")
    callbacks_list = list()

    
    logdir = 'logs/scalars/'+datetime.now().strftime('%Y%m%d-%H%M%S')
    # Keras Tensorboard callback

    tb_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks_list.append(tb_callback)
    print(callbacks_list)

    train_loader = DataGenerator(mode='train')
    val_loader = DataGenerator(mode='val')

    print('loaded data generators')
    model = siamese_architecture()

    print('training')
    train(n_epochs=args.n_epochs, lr=args.lr,
          model=model, train_gen=train_loader,
          val_gen=val_loader, callbacks_list=callbacks_list)


if __name__=='__main__':
    args = parser.parse_args()
    main(args)
