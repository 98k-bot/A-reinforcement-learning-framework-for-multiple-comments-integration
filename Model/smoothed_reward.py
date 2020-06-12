import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})

x1 = [1.85,1.88,1.89,1.865,1.855,1.869,1.865]
x2 = [1.85,1.863,1.845,1.826,1.820,1.815,1.819]
x3 = [1.863,1.867,1.84,1.836,1.834,1.83,1.835]
x4 = [1.84,1.842,1.843,1.825,1.818,1.82,1.819]
x5 = [1.79,1.794,1.774,1.763,1.761,1.763,1.762]
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
plt.savefig('1_cr.pdf', bbox_inches='tight')
plt.show()