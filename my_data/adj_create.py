import geopandas as gpd
import numpy as np

def create_adjacency_matrix(shp_path):
    # 读取shapefile文件
    gdf = gpd.read_file(shp_path)
    
    # 获取区域数量
    n = len(gdf)
    
    # 初始化邻接矩阵
    adj_matrix = np.zeros((n, n))
    
    # 构建邻接矩阵
    for i in range(n):
        for j in range(n):
            if i != j:
                # 如果两个多边形相邻（有共同边界），则设置为1
                if gdf.iloc[i].geometry.touches(gdf.iloc[j].geometry):
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
    
    # 保存邻接矩阵
    np.save('adjacency_matrix.npy', adj_matrix)
    return adj_matrix

# 使用示例
shp_path = "郑州社区.shp"  # 需要完整的shapefile文件
adj_matrix = create_adjacency_matrix(shp_path)
print("邻接矩阵形状:", adj_matrix.shape)