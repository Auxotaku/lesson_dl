from MakeDataset import MyDataset
from FCN import FCN
import torch
import torch.utils.data
import matplotlib.pyplot as plt

config = {
    "root": "./Dataset",
    "length": 1024,
    "batch_size": 128,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "num_classes": 10,
    "model_path": "./model.pth",
    "save_path": "./model.pth",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config["device"]), y.to(config["device"])
            output = model(x)
            pred = output.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
    return 100 * correct / total


if __name__ == '__main__':
    train_dataset = MyDataset(root=config["root"], length=config["length"], mode="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataset = MyDataset(root=config["root"], length=config["length"], mode="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"])
    model = FCN(input_length=config["length"], classes=config["num_classes"]).to(config["device"])
    print("Parameters Size:", sum(param.numel() for param in model.parameters()))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_acc, best_epoch = 0, 0
    loss_plot, acc_plot = [], []
    for epoch in range(config["num_epochs"]):
        loss_epoch = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(config["device"]), y.to(config["device"])

            output = model(x)
            loss = criterion(output, y)
            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), config["save_path"])

        loss_epoch /= len(train_loader)
        print(f"Epoch: {epoch+1}/{config['num_epochs']}")
        print(f"Loss: {loss_epoch:.8f}, Val Acc: {val_acc:.4f}%")
        print(f"Best Acc: {best_acc:.2f}%, Best Epoch: {best_epoch+1}")
        print("=====================================================")
        loss_plot.append(loss_epoch)
        acc_plot.append(val_acc)

    plt.plot(loss_plot)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.close()
    plt.plot(acc_plot)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('acc.png')
    plt.close()
