// Sankey diagram桑基图，可用于可视化数据流向的可视化方法

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 定义桑基图的节点和流量数据
labels = [
    "初始文献检索", 
    "排除标准(i)：会议海报", 
    "排除标准(ii)：重复发表", 
    "排除标准(iii)：不可访问", 
    "排除标准(iv)：方法结果描述不足", 
    "排除标准(v)：无新方法/结果", 
    "最终文献集"
]

source = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
target = [1, 2, 3, 4, 5, 6, 6, 6, 6, 6]
value = [149, 149, 149, 149, 149, 677-149-128-4-5, 128, 4, 5, 47]

# 创建桑基图数据
link = dict(source=source, target=target, value=value)
node = dict(label=labels, pad=15, thickness=20, line=dict(color="black", width=0.5),
            color="#808080")  # 移除 font 属性

# 构建桑基图
fig = go.Figure(data=[go.Sankey(
    node=node,
    link=link,
    arrangement="snap",
    hoverlabel=dict(
        bgcolor="white", 
        font=dict(family="SimHei", size=12)  # 悬停标签字体
    )
)])

# 设置图表标题和布局，包括全局字体
fig.update_layout(
    title_text="文献筛选流程桑基图",
    font=dict(family="SimHei", size=16),  # 全局字体设置
    width=800,
    height=600,
    margin=dict(t=50, b=50, l=100, r=100)
)

# 添加注释说明
fig.add_annotation(
    x=0.5, y=1.1,
    xref="paper", yref="paper",
    text="注：初始检索文献677篇，经5项排除标准筛选后剩余47篇",
    showarrow=False,
    font=dict(family="SimHei", size=12)
)

fig.show()
fig.write_html("literature_screening_sankey.html")
