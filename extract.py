import re

def parse_log_file(file_path):
    epochs = []
    training_losses = []
    validation_losses = []
    validation_accuracies = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        epoch_match = re.search(r'Epoch (\d+)/\d+, Training Loss: ([\d.]+)', line)
        val_loss_match = re.search(r'Validation Loss: ([\d.]+)', line)
        val_acc_match = re.search(r'Validation Accuracy: ([\d.]+)%', line)

        if epoch_match:
            epoch = int(epoch_match.group(1))
            training_loss = float(epoch_match.group(2))
            epochs.append(epoch)
            training_losses.append(training_loss)

        if val_loss_match:
            validation_loss = float(val_loss_match.group(1))
            validation_losses.append(validation_loss)

        if val_acc_match:
            validation_accuracy = float(val_acc_match.group(1))
            validation_accuracies.append(validation_accuracy)

    return epochs, training_losses, validation_losses, validation_accuracies

log_file_path = 'training.log'
epochs, training_losses, validation_losses, validation_accuracies = parse_log_file(log_file_path)
print(max(validation_accuracies))
