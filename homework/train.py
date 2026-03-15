import argparse
from .train_utils import train_classification, train_detection

TRAINER_FACTORY = {
    'classification': train_classification,
    'detector': train_detection
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_task", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    
    # load the training function
    training_task = args.training_task
    train_function = TRAINER_FACTORY[training_task]

    # delete the model name from the params add for training
    function_args = vars(parser.parse_args())
    del function_args['training_task']

    # train model
    train_function(**function_args)