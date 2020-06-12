import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})

x1 = [1.965,1.978,1.96,1.95,1.955,1.948,1.955]
x2 = [1.963,1.967,1.970,1.977,1.982,1.980,1.975]
x3 = [1.958,1.95,1.96,1.968,1.97,1.979,1.971]
x4 = [1.938,1.96,1.92,1.905,1.903,1.904,1.901]
x5 = [1.89,1.904,1.897,1.89,1.889,1.895,1.895]
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
plt.savefig('2_cr.pdf', bbox_inches='tight')
plt.show()