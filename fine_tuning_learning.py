import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.traning import train
from src.model import create_ft_model
from src.data_preparation import create_dataset, create_loader
from src.utils import TrainStoper, SaveController


if __name__ == "__main__":
    NUM_CLASSES = 43

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    train_dataset = create_dataset(path="data/train")
    val_dataset = create_dataset(path="data/val")
    train_loader = create_loader(train_dataset, batch_size=128, shuffle=True)
    val_loader = create_loader(val_dataset, batch_size=128, shuffle=False)

    model = create_ft_model(NUM_CLASSES).to(device)
    loss_func = nn.CrossEntropyLoss()
    opt = Adam([
            {"params":model.fc.parameters(), "weight_decay": 0.001, "lr": 0.001},
            {"params":model.layer4.parameters(), "weight_decay": 0.0001, "lr": 0.00003}
        ])
    lr_scheduler = ReduceLROnPlateau(
        optimizer=opt,
        mode = "min",
        factor=0.1,
        patience=5,
        threshold=0.0001,
        threshold_mode="rel"
    )
    # По val_loss
    train_stoper = TrainStoper(mode="min", patience=10, threshold=0.0001, threshold_mode="rel")
    save_controller = SaveController(mode="min")

    # ================== Обучение модели transfer learning ==================
    train(
        device=device, 
        train_dataset=train_dataset, val_dataset=val_dataset,
        train_loader=train_loader, val_loader=val_loader,
        model=model, num_classes=NUM_CLASSES,
        loss_func=loss_func, opt=opt, lr_scheduler=lr_scheduler,
        train_stoper=train_stoper, save_controller=save_controller,
        save_path="models/save_ft_models", EPOCHS=100
    )
