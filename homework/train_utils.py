from datetime import datetime
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .models import load_model, save_model
from .datasets.classification_dataset import load_data
from .metrics import ConfusionMatrix, AccuracyMetric, DetectionMetric

def train_detection(
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs
) -> None:


    raise NotImplementedError('detection not implemented')


def train_classification(
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs
) -> None:
    
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")

    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    
    # setting manual seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # setting the tensorboard logger
    log_dir = f'classifier/classifier_{datetime.now().strftime("%m%d_%H%M%S")}'
    logger = tb.SummaryWriter(log_dir)

    # load the model and place it in training mode
    model = load_model('classifier')
    model = model.to(device)
    model.train()

    # load the train and validation sets
    train_data = load_data('classification_data/train', shuffle=True, batch_size=batch_size)
    val_data = load_data('classification_data/val', batch_size=batch_size)

    # start the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)

    # start metric obj
    train_accuracy_obj = AccuracyMetric()
    val_accuracy_obj = AccuracyMetric()

    # training loop
    global_step = 0
    for epoch in range(num_epoch):
        
        # metric reset
        train_accuracy_obj.reset()
        val_accuracy_obj.reset()

        model.train()

        for sample, labels in train_data:

            sample, labels = sample.to(device), labels.to(device)

            # forward pass
            logits = model(sample)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            logger.add_scalar('train/loss', loss, global_step)

            # backpropogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate metrics
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            train_accuracy_obj.add(preds, labels)

            global_step += 1

        # validation accuracy
        model.eval()

        for sample, labels in val_data:

            sample, labels = sample.to(device), labels.to(device)

            logits = model(sample)
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            val_accuracy_obj.add(preds, labels)

        # train metrics
        train_metrics = train_accuracy_obj.compute()
        train_accuracy = train_metrics['accuracy']
        logger.add_scalar('train/acc', train_accuracy, global_step)

        # val metrics
        val_metrics = val_accuracy_obj.compute()
        val_accuracy = val_metrics['accuracy']
        logger.add_scalar('val/acc', val_accuracy, global_step)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={train_accuracy:.4f} "
                f"val_acc={val_accuracy:.4f}"
            )

    # save trained model here
    save_model(model)

    # save a copy to the log dir
    torch.save(model.state_dict(), log_dir + '/classifier.th')
    print(f'saved model to {log_dir}/classifier.th')
