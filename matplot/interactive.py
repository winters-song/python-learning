import numpy as np
from matplotlib import pyplot as plt
import time

plt.ion()
plt.plot([1.6, 2.7])
plt.pause(0.1)
time.sleep(2)

plt.title("interactive test")
plt.xlabel("index")

ax = plt.gca()
ax.cla()
ax.plot([3.1, 2.2])


plt.pause(0.1)
time.sleep(2)


ax = plt.gca()
ax.cla()
ax.plot([6, 2])
plt.pause(0)