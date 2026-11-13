import torch

loss = torch.load("./checkpoint/ClipPretrained/loss.pt", map_location=torch.device('cpu'))

import matplotlib.pyplot as plt

plt.plot(range(99), loss["training_loss"], color='r', label='Training Loss')
plt.plot(range(99), loss["validation_loss"], color='y', label='Validaiton Loss')
plt.title("Loss Curve (for Clip)")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.yscale('log')
plt.legend()
plt.show()
