import numpy as np
import scipy.sparse as ss
import pandas as pd

def load_od_data(data_path, data_size):
    """
    加载OD数据并进行预处理
    """
    # 生成日期序列
    OD_DAYS = [date.strftime('%Y-%m-%d') for date in 
               pd.date_range(start='2021-07-10', end='2021-07-31', freq='1D')]
    
    # 加载数据
    prov_day_data = ss.load_npz(data_path)
    prov_day_data_dense = np.array(prov_day_data.todense()).reshape(data_size)
    
    # 截取指定日期范围的数据并添加维度
    data = prov_day_data_dense[-len(OD_DAYS):, :, :, np.newaxis]
    
    # 对数变换
    ODdata = np.log(data + 1.0)
    
    print(f"数据形状: {ODdata.shape}")
    print(f"ODdata: {ODdata}")
    return ODdata

if __name__ == "__main__":
    file_path = r"C:\Users\steve\Desktop\xz\ODCRN\day_data\od_matrix.npz"
    file_path = r"C:\Users\steve\Desktop\xz\ODCRN\data\od_day20180101_20210228.npz"
    od_data = load_od_data(file_path,(-1,47,47))  