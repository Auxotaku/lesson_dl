from MakeDataset import MyDataset
from FCN import FCN
from TrainFrameFCN import config, evaluate
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

test_dataset = MyDataset(root=config["root"], length=config["length"], mode="val")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"])
model = FCN(input_length=config["length"], classes=config["num_classes"]).to(config["device"])
model.load_state_dict(torch.load(config["model_path"]))
model.eval()
test_acc = evaluate(model, test_loader)
print("Test Accuracy: {:.2f}%".format(test_acc))

# Visualize the confusion matrix
y_pred, y_true = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(config["device"]), y.to(config["device"])
        output = model(x)
        pred = output.argmax(dim=1)
        y_pred += pred.tolist()
        y_true += y.tolist()
cm = confusion_matrix(y_pred, y_true)
cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, annot_kws={"size": 12})
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.savefig('confusion_matrix.png')

