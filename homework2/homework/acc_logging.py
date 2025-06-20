from os import path
import torch
import torch.utils.tensorboard as tb


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    # start training
    global_step = 0
    #cumulative_train_accuracy = 0.0

    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        cumulative_train_accuracy = []
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            # Log the training loss every iteration
            train_logger.add_scalar('loss', dummy_train_loss, global_step)
            global_step += 1

            # Cumulative training accuracy for the epoch
            cumulative_train_accuracy.append(dummy_train_accuracy)

        # Calculate and log the average training accuracy for the epoch
        avg_train_accuracy = torch.mean(torch.stack(cumulative_train_accuracy)).item()
        train_logger.add_scalar('accuracy', avg_train_accuracy, global_step)

            
        torch.manual_seed(epoch)
        cumulative_validation_accuracy = []
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            cumulative_validation_accuracy.append(dummy_validation_accuracy)
        # Calculate and log the average validation accuracy for the epoch
        avg_valid_accuracy = torch.mean(torch.stack(cumulative_validation_accuracy)).item()
        valid_logger.add_scalar('accuracy', avg_valid_accuracy, global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
