import torch
from torch import Tensor


def calculate_segmentation_accuracy(predictions, labels):
    """
    Calculate pixel-wise accuracy for a segmentation model with multiple classes.

    Args:
        predictions (torch.Tensor): Model predictions (logits) of shape (batch_size, num_classes, height, width).
        labels (torch.Tensor): Ground truth labels of shape (batch_size, height, width).
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        float: Overall accuracy as a percentage.
    """
    # Ensure predictions are probabilities or logits, apply argmax along class dimension
    predicted_classes = torch.argmax(predictions, dim=1)  # Shape: (batch_size, height, width)
    predicted_classes = predicted_classes.view(-1)  # Shape: (total_pixels)
    labels = labels.view(-1)  # Shape: (total_pixels)
    correct_predictions = (predicted_classes == labels)

    # Calculate accuracy
    return correct_predictions.sum().item() / correct_predictions.numel()


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
