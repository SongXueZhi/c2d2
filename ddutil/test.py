import graph_tool as gt

g = gt.Graph()
# 添加节点和边
# ...
v1 = g.add_vertex()
v2 = g.add_vertex()


# 为边添加权重属性
weight_prop = g.new_edge_property("string")
e = g.add_edge(v1, v2,prop="sxz")
weight_prop[e] = "3.14" # 字符串形式的权重值

# 将图形保存为GraphML文件并保存边属性
g.save("graph.xml", fmt="graphml", edge_attrs=["weight"])
