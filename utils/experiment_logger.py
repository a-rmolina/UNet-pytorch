import logging
import json

import numpy as np
import multiprocessing as mp
import pandas as pd

from pathlib import Path
from skimage import io
from typing import Tuple, Union, Any
from collections import OrderedDict



def load_json(json_path: Union[str, Path]) -> np.ndarray[Any, np.dtype[Any]]:
    with open(json_path, 'r') as f:
        class_dict = OrderedDict(json.load(f))
        return np.array(list(class_dict.values()))


def build_color_map(labels_class_file: Path):
    # Load the colors in alphabetical order and transform them to a list.
    return load_json(labels_class_file)



def save_training_data(folder: Path, epoch: int, data: pd.DataFrame):
    out_file = folder / f"{epoch}.csv"
    mean_loss = data['loss'].mean()
    mean_acc = data['accuracy'].mean()
    mean_row = {'step': "Epoch Mean", 'loss': mean_loss, 'accuracy': mean_acc}
    data_to_save = pd.concat([pd.DataFrame([mean_row]), data], ignore_index=True)
    data_to_save.to_csv(out_file)


def setup_data_folder(output_folder: Path) -> Tuple[Path, Path]:
    if not output_folder.exists():
        output_folder.mkdir()
    training_logs_path = output_folder / 'training'
    training_logs_path.mkdir(exist_ok=True)
    evaluation_logs_path = output_folder / 'evaluation'
    evaluation_logs_path.mkdir(exist_ok=True)
    return training_logs_path, evaluation_logs_path


def add_new_training_data(training_data: pd.DataFrame, new_training_data: dict) -> pd.DataFrame:
    new_training_data.pop("epoch")
    new_row = pd.DataFrame([new_training_data])
    return new_row if training_data.empty else \
        pd.concat([training_data, new_row], ignore_index=True)


def save_evaluation_data(folder: Path, epoch: int, data: dict, data_keys: set):
    eval_results = {key: data[key] for key in data if key in data_keys}
    eval_data_path = folder / f"data_epoch_{epoch}.json"
    with open(eval_data_path, "w") as f:
        json.dump(eval_results, f)


def label_to_classes(labeled_image, label_classes):
    output = np.zeros((1, labeled_image.shape[0], labeled_image.shape[1], label_classes.shape[0]))

    for c, label_class in enumerate(label_classes):
        label = np.nanmin(label_class == labeled_image, axis=2)
        output[:, :, c] = label

    return output


def save_evaluation_images(folder: Path, epoch: int, data: dict, color_map: np.ndarray):
    eval_image_path = folder / f"image_epoch_{epoch}.jpg"

    image = data['images']*255
    true_mask = color_map[data["masks"]["true"]].astype(np.uint8)
    predicted_mask = color_map[data["masks"]["pred"]].astype(np.uint8)
    image = image.transpose(1, 2, 0).astype(np.uint8)
    eval_image_comparison = np.concatenate((image, true_mask, predicted_mask), axis=1)
    io.imsave(eval_image_path, eval_image_comparison)


def save_logs(stop: mp.Event, queue: mp.Queue, output_folder: Path, labels_path: Path):
    color_map = build_color_map(labels_path)
    training_logs_path, evaluation_logs_path = setup_data_folder(output_folder)
    eval_data_keys = {'learning_rate', 'validation_dice', 'step'}
    current_epoch = -1
    training_data = pd.DataFrame()
    while not stop.is_set() or not queue.empty():
        if not queue.empty():
            new_data: dict = queue.get()
            if "TRAINING" in new_data:
                if new_data["TRAINING"]["epoch"] != current_epoch:
                    if len(training_data) > 0:
                        save_training_data(training_logs_path, current_epoch, training_data)
                    current_epoch = new_data["TRAINING"]["epoch"]
                training_data = add_new_training_data(training_data, new_data["TRAINING"])

            elif "EVALUATION" in new_data:
                # Save Json with the results of the epoch
                eval_epoch = new_data["EVALUATION"].pop('epoch')
                save_evaluation_data(evaluation_logs_path, eval_epoch, new_data["EVALUATION"], eval_data_keys)
                save_evaluation_images(evaluation_logs_path, eval_epoch, new_data["EVALUATION"], color_map)
            else:
                logging.warning("Received dict can't be processed.")

    if len(training_data) > 0:
        save_training_data(training_logs_path, current_epoch, training_data)


class ExperimentLogger:
    def __init__(self):
        self.process = None
        self.queue = mp.Queue()
        self.stop = mp.Event()

    def start_saving(self, out_folder: Path, labels_path: Path):
        self.process = mp.Process(target=save_logs, args=(self.stop, self.queue, out_folder, labels_path), )
        self.process.start()

    def stop_saving(self):
        try:
            if self.process is not None:
                self.stop.set()
                self.process.join()
        except Exception as e:
            logging.error("Could not stop experiment logger")
            raise e

    def log(self, data: dict):
        if self.queue:
            self.queue.put(data)
