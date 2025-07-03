# Alluvial diagram冲积图，可用来进行趋势分析，一种可视化方法

import plotly.graph_objects as go
import pandas as pd
import warnings

# 忽略 pyarrow 缺失的警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Pyarrow will become a required dependency of pandas")

# 定义数据
data = pd.DataFrame({
    'source': ['本科-计算机', '本科-计算机', '本科-数学', '本科-数学', '本科-物理', '本科-物理'],
    'target': ['硕士-人工智能', '硕士-数据科学', '硕士-人工智能', '硕士-数据科学', '硕士-数据科学', '硕士-理论物理'],
    'value': [120, 80, 60, 90, 40, 70]
})

# 创建节点标签列表
nodes = list(set(data['source'].tolist() + data['target'].tolist()))

# 为每个节点分配索引
node_dict = {node: index for index, node in enumerate(nodes)}

# 准备桑基图所需的数据
source_indices = [node_dict[source] for source in data['source']]
target_indices = [node_dict[target] for target in data['target']]

# 创建桑基图
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color=["blue", "green", "purple", "orange", "red", "teal"]
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=data['value'],
        color='rgba(200, 200, 200, 0.4)'
    )
)])

# 设置图表标题
fig.update_layout(title_text="学生专业选择流向分析", font_size=12)

# 显示图表
fig.show()

# 若要保存为HTML文件，取消下面这行的注释
# fig.write_html("alluvial_diagram.html")
