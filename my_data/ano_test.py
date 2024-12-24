# import numpy as np
# data = np.load(r"C:\Users\steve\Desktop\xz\ODCRN\data\od_day20180101_20210228.npz")
# print(data.files)  # 输出: ['arr_0', 'my_array']
# print(data["indices"])  # 输出: ['arr_0', 'my_array']
# print(data["indptr"])  # 输出: ['arr_0', 'my_array']
# print(data["format"])  # 输出: ['arr_0', 'my_array']
# print(data["shape"])  # 输出: ['arr_0', 'my_array']
# print(data["data"])  # 输出: ['arr_0', 'my_array']

import numpy as np
from scipy.sparse import csc_matrix

data = np.load(r"C:\Users\steve\Desktop\xz\ODCRN\data\od_day20180101_20210228.npz")
data = np.load(r"C:\Users\steve\Desktop\xz\ODCRN\my_data\day_data\combined_od_data_2d.npz")

indices = data["indices"]
indptr = data["indptr"]
shape = data["shape"]
matrix_data = data["data"]

sparse_matrix = csc_matrix((matrix_data, indices, indptr), shape=shape)

print(sparse_matrix)  # 打印稀疏矩阵的表示形式

# 检查稀疏矩阵的属性
print(f"Shape: {sparse_matrix.shape}")
print(f"Data type: {sparse_matrix.dtype}")
print(f"Number of non-zero elements: {sparse_matrix.nnz}")
print(f"Density: {sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])}")