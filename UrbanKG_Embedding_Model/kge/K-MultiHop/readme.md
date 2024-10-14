## 关于如何生成多跳嵌入的结点关系矩阵（以芝加哥区域TransE嵌入为例）
### 1. 运行 gen_area_net_CHI.py 获得区域邻接关系矩阵 area_graph_CHI.pkl; 运行draw_area_net_CHI.py 用来绘制区域邻接关系图 area_net_CHI.png
### 2. 运行 kmh_info_CHI.py 获取结点之间的多跳嵌入关系 32_1hop.pkl, 其中 32 为嵌入维度, 1 为跳数
### 3. 配置 config.json ("ke_type": "CHIArea", "ke_model": "TransE"), 运行 MultiHop.py 获得结点间的相互关系 32_1hop.csv
### 4. 将生成的 32_1hop.pkl 与 32_1hop.csv 移动至 CHIArea/TransE 文件夹下

## tips
### 后续可以利用配置简化操作 4 