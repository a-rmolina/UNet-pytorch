import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from pathlib import Path


def test_experiment_logger(output_folder: Path):
    experiment = ExperimentLogger()
    labels_path = Path("/home/armolina/aaMain/workspace/multispectral_segmentation/SegNet_PyTorch/Postdam/postdam_classes.json")
    experiment.start_saving(output_folder, labels_path)
    global_step = 0
    img_size = 10
    loss = 1.0
    accuracy = 0.0
    val_score = 0.0
    for epoch in range(5):
        for _ in range(4):
            global_step += 1
            experiment.log({"TRAINING": {
                'step': global_step,
                'loss': loss,
                'accuracy': accuracy,
                'epoch': epoch
            }})
            accuracy += 0.05
            loss -= 0.05
        val_score += accuracy
        if epoch % 2 == 0:
            experiment.log({"EVALUATION": {
                'learning_rate': 0.005,
                'validation_dice': val_score,
                'images': np.ones((3, img_size, img_size), np.uint8) * val_score,
                'masks': {
                    'true': torch.tensor(np.ones((1, img_size, img_size)) * epoch).type(torch.long),
                    'pred': torch.tensor(np.ones((1, img_size, img_size)) * epoch + 1).type(torch.long),
                },
                'step': global_step,
                'epoch': epoch,
            }})

    experiment.stop_saving()


if __name__ == "__main__":
    out_folder = Path("test_output")
    test_experiment_logger(out_folder)
