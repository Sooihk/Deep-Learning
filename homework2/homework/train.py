from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
from os import path
import numpy as np

def train(args):
    from os import path
    # Checking to used CUDA as device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Intialize tensorboard summary writers if log directory is provided
    model = CNNClassifier().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    # construct the loss and optimizer 
    loss_function = torch.nn.CrossEntropyLoss()
    #loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum=args.momentum)

    # Load training and validation dataset to train the model
    train_dataset = load_data(args.train_data_path, num_workers=args.num_workers, batch_size=args.batch_size)
    validation_dataset = load_data(args.valid_data_path, num_workers=args.num_workers, batch_size=args.batch_size)

     # Training loop
    global_step = 0  # Initialize global step for logging
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        # Variables to track running loss and number of correct predictions during training
        running_loss = []
        training_acc = []
        validation_acc = []

        # Iterate over training dataset
        for inputs, labels in train_dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass to compute predictions
            outputs = model(inputs)
            # Compute loss and accuracy
            loss_value = loss_function(outputs, labels)
            accuracy_value = accuracy(outputs, labels)

            running_loss.append(loss_value.detach().cpu().numpy())
            training_acc.append(accuracy_value.detach().cpu().numpy())

            # log training loss
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_value, global_step=global_step)

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Backward pass to compute gradients with respect to the loss
            loss_value.backward()
            # update model parameters using current computed gradients
            optimizer.step()

            global_step += 1
        # Compute average training loss and accuracy for the epoch
        average_loss = sum(running_loss) / len(running_loss)
        train_accuracy = sum(training_acc) / len(training_acc)

        # Log average training accuracy for the epoch
        if train_logger is not None:
            train_logger.add_scalar('accuracy', np.mean(training_acc), global_step=global_step)
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Iterate over validation dataset
            for inputs, labels in validation_dataset:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute validation accuracy
                validation_acc.append(accuracy(outputs, labels).detach().cpu().numpy())

        # Compute validation accuracy for the epoch
        valid_accuracy = sum(validation_acc) / len(validation_acc)
        # log
        if valid_logger is not None:
            valid_logger.add_scalar('accuracy', np.mean(validation_acc), global_step=global_step)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Training Loss: {average_loss:.4f}, "
              f"Training Accuracy: {train_accuracy:.4f}, "
              f"Validation Accuracy: {valid_accuracy:.4f}")
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--train_data_path', type=str, default=r'data\train')
    parser.add_argument('--valid_data_path', type=str, default=r'data\valid')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    train(args)
