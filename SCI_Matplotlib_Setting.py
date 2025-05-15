import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# 替换为你的 Arial.ttf 路径
# font_path = "/mnt/data1/tanshiqian/.fonts/Arial.ttf"  # Linux
# font_path = "C:/Windows/Fonts/Arial.ttf"  # Windows
# font_path = "/Library/Fonts/Arial.ttf"  # macOS

# font_prop = FontProperties(fname=font_path, size=10)
# font_prop = FontProperties(fname=font_path)

# 设置全局参数
plt.rcParams.update({
    # 'font.sans-serif': 'Arial',
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.54, 2.5),  # 单栏矩形图
    'savefig.dpi': 600,
    'axes.titlesize': 10,
    # 'axes.labelweight': 'bold',
    # 'axes.titleweight': 'bold',
    # 'axes.titlepad': 5,
    # 'axes.labelpad': 5,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.5,
    'lines.markersize': 3,
    # 'lines.markeredgewidth': 0.5,
    # 'lines.markerfacecolor': 'black',
    # 'lines.markeredgecolor': 'black',
})
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.size'] = 10
# plt.rcParams['font.sans-serif'] = 'Arial'  # 设置全局字体

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.set_cmap('viridis')
