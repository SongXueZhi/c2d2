import networkx as nx
import matplotlib.pyplot as plt

# 这是一个假设的概率矩阵
prob_matrix = np.array([
    [0.0, 0.9, 0.1, 0.2],
    [0.9, 0.0, 0.2, 0.1],
    [0.1, 0.2, 0.0, 0.9],
    [0.2, 0.1, 0.9, 0.0]
])

# 计算距离矩阵
dist_matrix = 1.0 - prob_matrix

# 将距离矩阵转化为图
G = nx.from_numpy_array(dist_matrix)

# 使用 Prim 算法得到最大生成树
T = nx.minimum_spanning_tree(G)

# 可视化结果
nx.draw(T, with_labels=True)
plt.show()
