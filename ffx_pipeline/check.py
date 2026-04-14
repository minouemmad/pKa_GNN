import pickle, torch
data = pickle.load(open("Graph_pKa/Features/Datasets/data_list_0.pkl","rb"))[0]
print(data.x.shape)          # e.g. (10, 29)
print(data.edge_index.shape) # (2, n_edges)
print(data.edge_attr.shape)  # (n_edges, 4)  ← new
# edge_index and edge_attr must have same number of columns
assert data.edge_index.shape[1] == data.edge_attr.shape[0]