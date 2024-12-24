import os
import sys
import shutil
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch import optim
import Utils
from PDFormer import PDFormer

class ModelTrainer(object):
    def __init__(self, params:dict, data:dict, data_container):
        self.params = params
        self.data_container = data_container
        self.device = params['GPU']
        self.data = data  # 存储数据
        self.model = self.get_model(data).to(self.device)
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()

    def get_model(self, data):
        config = {
            'embed_dim': self.params['embed_dim'],
            'skip_dim': self.params['skip_dim'],
            'lape_dim': self.params['lape_dim'],
            'geo_num_heads': self.params['geo_num_heads'],
            'sem_num_heads': self.params['sem_num_heads'],
            't_num_heads': self.params['t_num_heads'],
            'mlp_ratio': self.params['mlp_ratio'],
            'qkv_bias': self.params['qkv_bias'],
            'drop': self.params['drop'],
            'attn_drop': self.params['attn_drop'],
            'drop_path': self.params['drop_path'],
            's_attn_size': self.params['s_attn_size'],
            't_attn_size': self.params['t_attn_size'],
            'enc_depth': self.params['enc_depth'],
            'type_ln': self.params['type_ln'],
            'type_short_path': self.params['type_short_path'],
            'output_dim': 1,
            'input_window': self.params['obs_len'],
            'output_window': self.params['pred_len'],
            'add_time_in_day': True,
            'add_day_in_week': True,
            'device': self.device,
            'world_size': 1,
            'huber_delta': 1,
            'quan_delta': 0.25,
            'far_mask_delta': 5,
            'dtw_delta': 5,
            'use_curriculum_learning': True,
            'step_size': 100,
            'max_epoch': self.params['num_epochs'],
            'task_level': 0
        }
        
        data_feature = {
            'num_nodes': self.params['N'],
            'feature_dim': 1,
            'ext_dim': 0,
            'num_batches': len(data['OD']) // self.params['batch_size'],
            'dtw_matrix': data['dtw_matrix'],  # 使用实际的DTW矩阵
            'adj_mx': data['adj'],
            'sd_mx': data['sd_mx'],  # 使用实际的距离矩阵
            'sh_mx': data['sh_mx'],  # 使用实际的跳数矩阵
            'pattern_keys': data['pattern_keys']  # 使用实际的模式键
        }
        
        return PDFormer(config, data_feature)

    def get_loss(self):
        if self.params['loss'] == 'MSE':
            criterion = nn.MSELoss(reduction='mean')
        elif self.params['loss'] == 'MAE':
            criterion = nn.L1Loss(reduction='mean')
        elif self.params['loss'] == 'Huber':
            criterion = nn.SmoothL1Loss(reduction='mean')
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'])
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer

    def train(self, data_loader:dict, modes:list, early_stop_patience=10):
        checkpoint = {'epoch': 0, 'state_dict': self.model.state_dict()}
        val_loss = np.inf
        patience_count = early_stop_patience

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     PDFormer model training begins:')
        for epoch in range(1, 1 + self.params['num_epochs']):
            starttime = datetime.now()
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                step = 0
                for x_seq, y_true in data_loader[mode]:
                    x_seq = x_seq.to(self.device)
                    y_true = y_true.to(self.device)
                    
                    # 确保pattern_keys的维度正确
                    pattern_keys = self.data['pattern_keys']  # 从self.data中获取pattern_keys
                    if pattern_keys.shape[-1] != 1:
                        pattern_keys = pattern_keys[..., np.newaxis]
                    pattern_keys = torch.FloatTensor(pattern_keys).to(self.device)
                    
                    # 打印维度信息
                    print(f"x_seq shape: {x_seq.shape}")
                    print(f"y_true shape: {y_true.shape}")
                    print(f"pattern_keys shape: {pattern_keys.shape}")
                    
                    batch = {
                        'X': x_seq,
                        'y': y_true,
                        'pattern_keys': pattern_keys
                    }
                    
                    with torch.set_grad_enabled(mode=(mode=='train')):
                        if mode == 'train':
                            self.optimizer.zero_grad()
                        
                        loss = self.model.calculate_loss(batch, step)
                        
                        if mode == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss[mode] += loss.item() * y_true.shape[0]
                    step += y_true.shape[0]
                    torch.cuda.empty_cache()

                if mode == 'validate':
                    epoch_val_loss = running_loss[mode]/step
                    if epoch_val_loss <= val_loss:
                        print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                              f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s')
                        val_loss = epoch_val_loss
                        checkpoint.update(epoch=epoch, state_dict=self.model.state_dict())
                        torch.save(checkpoint, self.params['output_dir']+f'/PDFormer_od.pkl')
                        patience_count = early_stop_patience
                    else:
                        print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.', f'used {(datetime.now() - starttime).seconds}s')
                        patience_count -= 1
                        if patience_count == 0:
                            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                            print(f'    Early stopping at epoch {epoch}. PDFormer model training ends.')
                            return   

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     PDFormer model training ends.')
        torch.save(checkpoint, self.params['output_dir']+f'/PDFormer_od.pkl')
        return

    def test(self, data_loader:dict, modes:list):
        trained_checkpoint = torch.load(self.params['output_dir']+f'/PDFormer_od.pkl')
        self.model.load_state_dict(trained_checkpoint['state_dict'])
        self.model.eval()

        for mode in modes:
            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            print(f'     PDFormer model testing on {mode} data begins:')
            forecast, ground_truth = [], []
            
            for x_seq, y_true in data_loader[mode]:
                batch = {
                    'X': x_seq,
                    'y': y_true
                }
                y_pred = self.model.predict(batch)
                
                forecast.append(y_pred.cpu().detach().numpy())
                ground_truth.append(y_true.cpu().detach().numpy())

            forecast = np.concatenate(forecast, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            if mode == 'test':
                np.save(self.params['output_dir'] + '/PDFormer_prediction.npy', forecast)
                np.save(self.params['output_dir'] + '/PDFormer_groundtruth.npy', ground_truth)
            
            # evaluate on metrics
            MSE, RMSE, MAE, MAPE = self.evaluate(forecast, ground_truth)
            f = open(self.params['output_dir'] + '/PDFormer_prediction_scores.txt', 'a')
            f.write("%s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (mode, MSE, RMSE, MAE, MAPE))
            f.close()

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     PDFormer model testing ends.')
        return
    
    @staticmethod
    def evaluate(y_pred: np.array, y_true: np.array, precision=10):
        def MSE(y_pred: np.array, y_true: np.array):
            return np.mean(np.square(y_pred - y_true))
        def RMSE(y_pred:np.array, y_true:np.array):
            return np.sqrt(MSE(y_pred, y_true))
        def MAE(y_pred:np.array, y_true:np.array):
            return np.mean(np.abs(y_pred - y_true))
        def MAPE(y_pred:np.array, y_true:np.array, epsilon=1e-0):
            return np.mean(np.abs(y_pred - y_true) / (y_true + epsilon))
        
        print('MSE:', round(MSE(y_pred, y_true), precision))
        print('RMSE:', round(RMSE(y_pred, y_true), precision))
        print('MAE:', round(MAE(y_pred, y_true), precision))
        print('MAPE:', round(MAPE(y_pred, y_true)*100, precision), '%')
        return MSE(y_pred, y_true), RMSE(y_pred, y_true), MAE(y_pred, y_true), MAPE(y_pred, y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OD Prediction with PDFormer')

    # command line arguments
    parser.add_argument('-GPU', '--GPU', type=str, help='Specify GPU usage', default='cuda:0')
    parser.add_argument('-in', '--input_dir', type=str, default='../../data')
    parser.add_argument('-out', '--output_dir', type=str, default='./output')
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=12)
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=12)
    parser.add_argument('-split', '--split_ratio', type=float, nargs='+',
                        help='Relative data split ratio in train : validate : test'
                             ' Example: -split 6.4 1.6 2', default=[6.4, 1.6, 2])
    parser.add_argument('-batch', '--batch_size', type=int, default=32)
    
    # PDFormer specific parameters
    parser.add_argument('-embed', '--embed_dim', type=int, default=64)
    parser.add_argument('-skip', '--skip_dim', type=int, default=256)
    parser.add_argument('-lape', '--lape_dim', type=int, default=8)
    parser.add_argument('-geo_heads', '--geo_num_heads', type=int, default=4)
    parser.add_argument('-sem_heads', '--sem_num_heads', type=int, default=2)
    parser.add_argument('-t_heads', '--t_num_heads', type=int, default=2)
    parser.add_argument('-mlp', '--mlp_ratio', type=float, default=4.0)
    parser.add_argument('-qkv', '--qkv_bias', type=bool, default=True)
    parser.add_argument('-drop', '--drop', type=float, default=0.0)
    parser.add_argument('-attn_drop', '--attn_drop', type=float, default=0.0)
    parser.add_argument('-path_drop', '--drop_path', type=float, default=0.3)
    parser.add_argument('-s_attn', '--s_attn_size', type=int, default=3)
    parser.add_argument('-t_attn', '--t_attn_size', type=int, default=3)
    parser.add_argument('-depth', '--enc_depth', type=int, default=6)
    parser.add_argument('-ln', '--type_ln', type=str, default='pre')
    parser.add_argument('-path', '--type_short_path', type=str, default='hop')
    
    parser.add_argument('-epoch', '--num_epochs', type=int, default=200)
    parser.add_argument('-loss', '--loss', type=str, help='Specify loss function',
                        choices=['MSE', 'MAE', 'Huber'], default='MSE')
    parser.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', default='Adam')
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3)
    parser.add_argument('-test', '--test_only', type=int, default=0)

    params = parser.parse_args().__dict__

    # paths
    os.makedirs(params['output_dir'], exist_ok=True)

    # load data
    data_input = Utils.DataInput(data_dir=params['input_dir'])
    data = data_input.load_data()
    params['N'] = data['OD'].shape[1]

    # get data loader
    data_generator = Utils.DataGenerator(obs_len=params['obs_len'], 
                                         pred_len=params['pred_len'], 
                                         data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(data=data, params=params)

    # get model
    trainer = ModelTrainer(params=params, data=data, data_container=data_input)
    
    if bool(params['test_only']) == False:
        trainer.train(data_loader=data_loader,
                      modes=['train', 'validate'])
    trainer.test(data_loader=data_loader,
                 modes=['train', 'test']) 