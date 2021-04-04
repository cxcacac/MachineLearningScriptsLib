import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# import os

deformation = pd.read_csv("./deformation.csv", encoding='utf-8')
moment = pd.read_csv("./bendingmoment.csv", encoding='utf-8')
shear = pd.read_csv("./shear.csv", encoding='utf-8')
compare = pd.read_csv("./deform1.csv", encoding='utf-8')

length = deformation['l']

l = np.array(length)
d1 = np.array(deformation.iloc[:, 1:])
m = np.array(moment.iloc[:, 1:])
s = np.array(shear.iloc[:, 1:])
c = np.array(compare.iloc[:, 1:])

d = s # HTPERPARAMETER

figsize = 9, 5
fig, ax = plt.subplots(figsize=figsize)
color = ['blue', 'green', 'black', 'orange']
# label = ['K=0 kN/m', 'K=50000 kN/m', 'K=100000 kN/m', 'Pinned']
label = ['K=0 kN/m', "K=50000 kN/m", 'K=100000 kN/m', 'Pinned']
# # marker = ['x', '+', '*', '.']
for i in range(d.shape[1]):
    data = d[:, i]/1000
    ax.plot(l, data, color=color[i], label=label[i], linewidth=1.5)

# 接下来就是修饰图
# 图例的字体设置
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
plt.legend(prop=font1)

# tick, major_locator确定刻度分隔，刻度的字体大小和字体类型。
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.tick_params(labelsize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
plt.xlabel('Beam Length (m)', font2)
plt.ylabel('deformation(mm)', font2)

# 一般情况下不需要title
# plt.title('Deformation shape with different spring support stiffness', font2)
# ax.grid(True)

# 设置x轴和y轴的显示范围
plt.xlim(0, 11)
# plt.ylim(-100000, 100000)
plt.savefig('shear.jpg')
plt.show()


