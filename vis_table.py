import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi

# 1. 准备数据 (去掉 Overall)
data = {
    "Metric": ["BERTScore", "ROUGE", "METEOR", "BLEU", "OE-Eval"],
    "QwenVL2.5-72B": [61.16, 67.04, 69.00, 59.32, 68.31],
    "Gemini-2.5-Flash": [60.34, 66.90, 67.12, 58.96, 67.13],
    "GPT-4o": [63.06, 70.01, 67.34, 60.57, 69.85],
    "InternVL3-78B": [59.79, 63.29, 64.55, 57.14, 63.99]
}
df = pd.DataFrame(data)

# 2. 设置雷达图的维度（以模型为轴）
categories = list(df.columns[1:]) # ['QwenVL2.5-72B', 'Gemini-2.5-Flash', 'GPT-4o', 'InternVL3-78B']
N = len(categories)

# 计算角度
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] # 闭合回路

# 初始化绘图
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 设置方向（顺时针或逆时针，这里设为正北为0度）
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# 设置轴标签
plt.xticks(angles[:-1], categories, size=11)

# 设置Y轴刻度（因为数据在55-70之间，如果不截断Y轴，差异会看不出来）
ax.set_rlabel_position(0)
plt.yticks([58, 62, 66, 70], ["58", "62", "66", "70"], color="grey", size=9)
plt.ylim(56, 71) # 关键：设置下限为56，放大差异

# 3. 绘制每个 Metric 的线条
# 策略：将 OE-Eval 单独拿出来最后画（加粗、高亮），其他作为背景（灰色/细线）

# 先画背景指标 (非 OE-Eval)
colors = ['#b0bec5', '#90a4ae', '#78909c', '#607d8b'] # 灰色系
line_styles = [':', '--', '-.', ':'] # 不同虚线样式区分

other_metrics = df[df['Metric'] != 'OE-Eval']
for idx, row in other_metrics.iterrows():
    values = row[categories].tolist()
    values += values[:1] # 闭合
    ax.plot(angles, values, linewidth=1.5, linestyle=line_styles[idx % 4], 
            label=row['Metric'], color='gray', alpha=0.6)

# 最后画主角 OE-Eval (高亮)
oe_eval_row = df[df['Metric'] == 'OE-Eval'].iloc[0]
values = oe_eval_row[categories].tolist()
values += values[:1]
ax.plot(angles, values, linewidth=3, linestyle='solid', label='OE-Eval (Ours)', color='#d62728') # 红色
ax.fill(angles, values, 'r', alpha=0.1) # 轻微填充

# 添加图例
# 将图例放在图外，避免遮挡
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)

plt.tight_layout()

# 保存
plt.savefig("radar_oe_eval.pdf", bbox_inches='tight')
plt.show()
