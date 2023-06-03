import matplotlib.pyplot as plt
import torch

x = torch.arange(0, 2, 0.1)

plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')
#
# plt.xlabel('x label')
# plt.ylabel('y label')
#
# plt.title("Simple Plot")
#
# plt.legend()

plt.show()