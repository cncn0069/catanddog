import read
import matplotlib.pyplot as plt

acc = read.history.history['acc']
val_acc = read.history.history['val_acc']
loss = read.history.history['loss']
val_loss = read.history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Traing loss')