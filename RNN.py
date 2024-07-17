import torch
import matplotlib.pyplot as plt
from torch import nn

class Rnn(nn.Module):
    def __init__(self, input_size):
        super(Rnn, self).__init__()
        self.RNN = nn.RNN(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_0):
        r_out, h_n = self.RNN(x, h_0)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_n

def generate_data(batch_size, time_steps, input_size):
    inputs = torch.randn(batch_size, time_steps, input_size)
    targets = torch.randn(batch_size, time_steps, 1)
    return inputs, targets

def train(model, data_loader, optimizer, criterion, epochs):
    losses = []
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            h_0 = torch.zeros(1, inputs.size(0), 32)  # 初始化隐藏状态
            optimizer.zero_grad()
            outputs, _ = model(inputs, h_0)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return losses

TIME_STEP = 10
INPUT_SIZE = 1
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.02

model = Rnn(INPUT_SIZE)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# generator data
data_loader = [generate_data(BATCH_SIZE, TIME_STEP, INPUT_SIZE) for _ in range(EPOCHS)]

#train(model, data_loader, optimizer, criterion, EPOCHS)
losses = train(model, data_loader, optimizer, criterion, EPOCHS)

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()