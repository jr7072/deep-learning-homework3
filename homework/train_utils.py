from datetime import datetime
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .models import load_model, save_model
from .datasets.classification_dataset import load_data as load_classification_data
from .datasets.road_dataset import load_data as load_drive_data
from .metrics import AccuracyMetric, DetectionMetric
from .loss import MultiClassFocalLoss

def get_device() -> torch.DeviceObjType:
    '''
        loads the device to use for training
    '''

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")

    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    return device


def train_detection(
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs
) -> None:

    device = get_device()

    # set manual seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set the log dir
    log_dir = f'detector/detector_{datetime.now().strftime("%m%d_%H%M%S")}'
    logger = tb.SummaryWriter(log_dir)

    # load the model
    model = load_model('detector')
    model = model.to(device)
    model.train()

    # load the data
    print("loading drive data")
    train_data = load_drive_data('drive_data/train', shuffle=True, batch_size=batch_size)
    val_data = load_drive_data('drive_data/val', batch_size=batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    
    # create metric obj here
    train_metric = DetectionMetric()
    val_metric = DetectionMetric()

    # define focal loss
    mcf_loss = MultiClassFocalLoss(alpha=torch.tensor([.6, 5, 5]), gamma=3).to(device)
    
    print(f'started training loop')
    # training loop
    global_step = 0
    for epoch in range(num_epoch):

        # reset the metric here
        train_metric.reset()
        val_metric.reset()
        
        model.train()
        # pass through the training data batch
        for data in train_data:
            
            images = data['image'].to(device)
            depth_labels = data['depth'].to(device)
            track_labels = data['track'].to(device)

            track_logits, depth_pred = model(images)

            # capture the accuracy here
            track_pred = track_logits.argmax(dim=1)
            train_metric.add(
                track_pred,
                track_labels,
                depth_pred,
                depth_labels
            )

            # backpropogate with a combined loss
            track_loss = mcf_loss(
                track_logits,
                track_labels
            )

            depth_loss = torch.nn.functional.mse_loss(
                depth_pred,
                depth_labels
            )

            # combine the losses
            loss = track_loss + depth_loss

            logger.add_scalar(
                'train/loss',
                loss,
                global_step=global_step
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # validation pass
        model.eval()
        for data in val_data:
            
            images = data['image'].to(device)
            depth_labels = data['depth'].to(device)
            track_labels = data['track'].to(device)

            track_logits, depth_pred = model(images)
            
            # capture the accuracy here
            track_pred = track_logits.argmax(dim=1)

            # backpropogate with a combined loss
            track_loss = mcf_loss(
                track_logits,
                track_labels
            )

            depth_loss = torch.nn.functional.mse_loss(
                depth_pred,
                depth_labels
            )

            # combine the losses
            loss = track_loss + depth_loss

            # capture validation loss
            logger.add_scalar('val/loss', loss, global_step=global_step)
            
            # capture the accuracy here
            val_metric.add(
                track_pred,
                track_labels,
                depth_pred,
                depth_labels
            )

        # calculate metrics
        train_results = train_metric.compute()
        val_results = val_metric.compute()

        # record train metrics
        train_iou = train_results['iou']
        train_abs_depth_err = train_results['abs_depth_error']
        train_tp_depth_err = train_results['tp_depth_error']

        logger.add_scalar(
            'train/iou',
            train_iou,
            global_step=global_step
        )
        logger.add_scalar(
            'train/abs_depth_err',
            train_abs_depth_err,
            global_step=global_step
        )
        logger.add_scalar(
            'train/tp_depth_err',
            train_tp_depth_err,
            global_step=global_step
        )

        # record val metrics
        val_iou = val_results['iou']
        val_abs_depth_err = val_results['abs_depth_error']
        val_tp_depth_err = val_results['tp_depth_error']

        logger.add_scalar(
            'val/iou',
            val_iou,
            global_step=global_step
        )
        logger.add_scalar(
            'val/abs_depth_err',
            val_abs_depth_err,
            global_step=global_step
        )
        logger.add_scalar(
            'val/tp_depth_err',
            val_tp_depth_err,
            global_step=global_step
        )

        logger.add_scalar(
            'meta/epoch',
            epoch,
            global_step=global_step
        )

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}:\n"
                f"\ttrain_iou={train_iou:.4f} "
                f"val_iou={val_iou:.4f}\n"
                f"\ttrain_abs_depth_error={train_abs_depth_err:.4} "
                f"val_abs_depth_error={val_abs_depth_err:.4}\n"
                f"\ttrain_tp_depth_error={train_tp_depth_err:.4} "
                f"val_tp_depth_error={val_tp_depth_err:.4}"
            )

    # save trained model here
    save_model(model)

    # save a copy to the log dir
    torch.save(model.state_dict(), log_dir + '/detector.th')
    print(f'saved model to {log_dir}/detector.th')


def train_classification(
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs
) -> None:
    
    device = get_device()
    
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
    train_data = load_classification_data('classification_data/train', shuffle=True, batch_size=batch_size, transform_pipeline='aug')
    val_data = load_classification_data('classification_data/val', batch_size=batch_size)

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
