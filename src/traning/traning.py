import torch
from tqdm import tqdm
from ..utils import save_model


def train(
        device,
        train_dataset, val_dataset,
        train_loader, val_loader,
        model, num_classes,
        loss_func, opt, lr_scheduler,
        train_stoper, save_controller,
        save_path, EPOCHS,
        start_epochs=0, 
        history_of_education = None
    ):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    lr_list = []

    for epoch in range(start_epochs, EPOCHS):
        train_loss_run = []
        val_loss_run = []

        true_train_answer = 0
        true_val_answer = 0

        mean_train_loss = None
        mean_val_loss = None

        run_train_acc = None
        run_val_acc = None

        train_loop = tqdm(train_loader, leave=False)
        model.train()
        for sample, target in train_loop:
            target = torch.eye(num_classes)[target].to(device)
            pred = model(sample.to(device))
            loss = loss_func(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_run.append(loss.item())
            mean_train_loss = sum(train_loss_run)/len(train_loss_run) 
            true_train_answer += (pred.argmax(dim=1) == target.argmax(dim=1)).sum().item()
            
            train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]; train_loss={mean_train_loss:.4f}")

        train_loss.append(mean_train_loss)
        run_train_acc = true_train_answer/len(train_dataset)
        train_acc.append(run_train_acc)

        val_loop = tqdm(val_loader, leave=False)
        model.eval()
        with torch.no_grad():
            for sample, target in val_loop:
                target = torch.eye(num_classes)[target].to(device)
                pred = model(sample.to(device))

                loss = loss_func(pred, target)

                val_loss_run.append(loss.item())
                mean_val_loss = sum(val_loss_run)/len(val_loss_run)
                true_val_answer += (pred.argmax(dim=1) == target.argmax(dim=1)).sum().item()

                val_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]; val_loss={mean_val_loss:.4f}")

        val_loss.append(mean_val_loss)
        run_val_acc = true_val_answer/len(val_dataset)
        val_acc.append(run_val_acc)
        lr_list.append(lr_scheduler.get_last_lr())

        print(f"Epoch [{epoch+1}/{EPOCHS}]; train_loss={mean_train_loss:.4f}; train_acc={run_train_acc:.4f};\
            val_loss={mean_val_loss:.4f}; val_acc={run_val_acc:.4f}; lr={lr_scheduler.get_last_lr()}")

        if save_controller(mean_val_loss):
            save_model(
                epoch=epoch+1, epochs=EPOCHS,
                model_state_dict=model.state_dict(),
                loss_func=loss_func,
                optimizer_state_dict=opt.state_dict(),
                lr_scheduler_state_dict=lr_scheduler.state_dict(),
                train_stoper=train_stoper, save_controller=save_controller,
                train_loss=train_loss, val_loss=val_loss,
                train_acc=train_acc, val_acc=val_acc, lr_list=lr_list,
                save_path=save_path
            )
            print(f"Epoch [{epoch+1}/{EPOCHS}] === MODEL SAVE ===")

        if train_stoper(mean_val_loss):
            print(f"Epoch [{epoch+1}/{EPOCHS}] !!! TRANING STOPPED !!!")
            break

        lr_scheduler.step(mean_val_loss)

    print("END")
