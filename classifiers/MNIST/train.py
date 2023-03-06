import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataset, check_accuracy
from model import NN

BATCH_SIZE = 64
NUM_EPOCHS = 2
LEARNING_RATE = 0.005

train_loader, test_loader = get_dataset(BATCH_SIZE)

model = NN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
total_step = len(train_loader)

for epoch in range(NUM_EPOCHS):
    losses = []

    for i, (data, targets) in enumerate(train_loader):
        outputs = model(data)
        loss = criterion(outputs, targets)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print (f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{total_step}], Loss: {loss.item()}")

torch.save(model.state_dict(),"model.pt")
print(f'train: {check_accuracy(train_loader, model)}')
print(f'test: {check_accuracy(test_loader, model)}')
            