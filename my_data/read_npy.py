import numpy as np

file_path = './adjacency_matrix.npy'

data = np.load(file_path)
# 打印文件的基本信息
print("Shape of the array:", data.shape)
print("Data type:", data.dtype)
# 打印数组的部分内容
print("First few elements of the array:")
print(data)  
# 如果你想要查看特定位置的数据，可以使用索引
print("\nElement at position (0, 0):")
print(data[0, 0])
# 如果数组是多维的，你可以打印更多维度的数据
if len(data.shape) > 1:
    print("\nFirst few elements of the first row:")
    print(data[0, :5])  # 显示第一行的前五个元素