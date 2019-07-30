from model import siamese_architecture
from dataloader import DataGenerator

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=0.001, type=float, help='initial LR')
parser.add_argument('--n_epochs', default=10, type=int, help='Epochs to train')
parser.add_argument('--b', default=32, type=int, help='Batch Size')


def main(args):
       
    train_loader = DataGenerator(mode='train')
    val_loader = DataGenerator(mode='val')

    model = siamese_architecture()

    model.fit_generator(generator=train_loader,
                        validation_data=val_loader,
                        epochs=args.n_epochs)

if __name__=='__main__':
    args = parser.parse_args()
    main(args)
