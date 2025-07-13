import matplotlib.pyplot as plt

# Data extracted manually from screenshots

epochs = list(range(10))

# Batch 32 (label 32-bs)
loss_32  = [7.216, 6.304, 6.232, 6.140, 6.128, 6.070, 6.051, 6.039, 5.995, 5.975]
sisdr_32 = [-14.37, -13.85, -13.75, -13.71, -13.69, -13.67, -13.67, -13.67, -13.66, -13.65]

# Batch 64 (label 64-bs)
loss_64  = [7.568, 7.215, 6.449, 6.291, 6.226, 6.148, 6.095, 6.119, 6.173, 6.060]
sisdr_64 = [-14.63, -14.35, -13.98, -13.84, -13.78, -13.74, -13.72, -13.70, -13.69, -13.69]

# Batch 100 (label 100-bs)
loss_100  = [7.588, 7.572, 7.176, 6.739, 6.444, 6.247, 6.208, 6.173, 6.161, 6.161]
sisdr_100 = [-14.66, -14.59, -14.39, -14.09, -13.95, -13.85, -13.80, -13.77, -13.75, -13.73]

# Graph 1: Loss
plt.figure(figsize=(6,4))
plt.plot(epochs, loss_32,  marker='o', label='32-bs')
plt.plot(epochs, loss_64,  marker='o', label='64-bs')
plt.plot(epochs, loss_100, marker='o', label='100-bs')
plt.xlabel('Epoch')
plt.ylabel('Training Loss (l)')
plt.title('Training Loss vs Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Graph 2: SI-SDR
plt.figure(figsize=(6,4))
plt.plot(epochs, sisdr_32,  marker='o', label='32-bs')
plt.plot(epochs, sisdr_64,  marker='o', label='64-bs')
plt.plot(epochs, sisdr_100, marker='o', label='100-bs')
plt.xlabel('Epoch')
plt.ylabel('Dev SI-SDR (dB)')
plt.title('Dev SI-SDR vs Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
