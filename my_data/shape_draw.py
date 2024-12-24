import geopandas as gpd
import matplotlib.pyplot as plt

def plot_shapefile(shp_path, output_path='map.png'):
    # 读取shapefile
    gdf = gpd.read_file(shp_path)
    
    # 创建图形
    fig, ax = plt.figure(figsize=(15, 15)), plt.gca()
    
    # 绘制地图
    gdf.plot(ax=ax, edgecolor='black', facecolor='none')
    
    # 添加区域标签（如果需要）
    # for idx, row in gdf.iterrows():
    #     # 获取多边形的中心点
    #     centroid = row.geometry.centroid
    #     # 如果有名称字段，使用实际的名称字段
    #     if 'name' in gdf.columns:
    #         label = row['name']
    #     else:
    #         label = f'Region {idx}'
    #     ax.annotate(label, (centroid.x, centroid.y), ha='center', va='center')
    
    # 设置标题
    plt.title('郑州社区地图')
    
    # 去除坐标轴
    plt.axis('off')
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"地图已保存为: {output_path}")
    
    # 打印一些基本信息
    print("\n地图信息:")
    print(f"总区域数量: {len(gdf)}")
    print("\n数据列:")
    print(gdf.columns.tolist())
    
    return gdf

# 使用示例
shp_path = "郑州社区.shp"
gdf = plot_shapefile(shp_path)