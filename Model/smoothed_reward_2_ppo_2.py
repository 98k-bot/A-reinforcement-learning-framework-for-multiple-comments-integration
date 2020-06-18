import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})

x1 = [3.963,3.975,3.97,3.96,3.965,3.968,3.965]
x2 = [3.963,3.968,3.970,3.982,3.979,3.980,3.975]
x3 = [3.958,3.959,3.968,3.963,3.965,3.969,3.971]
x4 = [3.937,3.965,3.938,3.935,3.935,3.934,3.931]
x5 = [3.894,3.914,3.896,3.899,3.898,3.895,3.897]
y1 = [10,20,30,40,50,60,70]

fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
line1, = ax.plot(y1, x1, label='f1')
plt.scatter(y1,x1)
line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line2, = ax.plot(y1, x2, label='f2')
plt.scatter(y1,x2)
line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line3, = ax.plot(y1, x3, label='f3')
plt.scatter(y1,x3)
line3.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line4, = ax.plot(y1, x4, label='f4')
plt.scatter(y1,x4)
line4.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
line5, = ax.plot(y1, x5, label='f5')
plt.scatter(y1,x5)
line5.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
plt.xlabel("Epoch")
plt.ylabel("Smoothed Reward on Each Fold")
# Using plot(..., dashes=...) to set the dashing when creating a line
#line2, = ax.plot(x, y - 0.2, dashes=[6, 2], label='Using the dashes parameter')

ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('2_r.png', bbox_inches='tight')
plt.show()