import torch
import os

class SaveController:
    def __init__(self, mode="min", threshold=0, threshold_mode="rel"):
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None

    def calculation(self, value):
        if self.mode == "min" and self.threshold_mode == "abs":
            return value < self.best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "abs":
            return value > self.best + self.threshold
        elif self.mode == "min" and self.threshold_mode == "rel":
            return value < self.best - self.best * self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return value > self.best + self.best * self.threshold
        
    def __call__(self, value):
        if self.best is None:
            self.best = value
            return True
        
        if self.calculation(value):
            self.best = value
            return True
        
        return False
        

def save_model(
        epoch, epochs, model_state_dict, loss_func,
        optimizer_state_dict, lr_scheduler_state_dict,
        train_stoper, save_controller,
        train_loss, val_loss, train_acc, val_acc,
        lr_list, save_path
    ):
    checkpoint = {
        "epoch": f"{epoch}/{epochs}",
        "model_state_dict": model_state_dict,
        "loos_func": loss_func,
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
        "train_stoper": train_stoper,
        "save_controller": save_controller,
        "history_of_education": {
            "loss": {
                "train": train_loss,
                "val": val_loss
            },
            "accuracy": {
                "train": train_acc,
                "val": val_acc
            },
            "lr_list": lr_list
        }
    }
    torch.save(checkpoint, os.path.join(save_path, f"my_resnet50_{epoch}.pth"))