import torch

class FCN(torch.nn.Module):
    def __init__(self, input_length, classes=10):
        super(FCN, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(input_length, int(1.3 * input_length)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(1.3 * input_length), int(0.7 * input_length)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(0.7 * input_length), int(0.3 * input_length)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(0.3 * input_length), classes),
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        out = self.softmax(out)
        return out
