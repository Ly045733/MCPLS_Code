import torch
import os
import random

import numpy as np
import pandas as pd 

from tqdm import tqdm
from networks.net_factory import net_factory
from dataloaders.dataset import BaseDataSets, RandomGenerator
from torchvision import transforms
from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn
from scipy.spatial.distance import directed_hausdorff
from medpy import metric

from openpyxl import load_workbook


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)
def hausdorff_distance95(y_true, y_pred):

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    # print(np.unique(y_pred_np))
    if np.sum(y_pred_np) == 0:
        return 0.0
    
    
    
    hd95 = metric.binary.hd95(y_pred_np, y_true_np)
    


    
    return hd95

def recall_score(y_true, y_pred, smooth=1e-6):
    # 计算真正例（TP）和假负例（FN）
    TP = torch.sum((y_true == 1) & (y_pred == 1))
    FN = torch.sum((y_true == 1) & (y_pred == 0))
    return (TP + smooth) / (TP + FN + smooth)  # 添加小的常数避免除以零

def calculate_metrics(y_true, y_pred, num_classes):
    dices = []
    HD95  = []
    # print(np.unique(y_true.cpu().detach().numpy()))
    # print(np.unique(y_pred.cpu().detach().numpy()))
    for cls in range(1,num_classes):
        cls_y_true = (y_true == cls).long()
        cls_y_pred = (y_pred == cls).long()

        dices.append(dice_coefficient(cls_y_true, cls_y_pred))
        HD95.append(hausdorff_distance95(cls_y_true,cls_y_pred))

    return dices,HD95


def val(root_path, fold, patch_size, num_classes, net_type, model_path):
    model = net_factory(net_type=net_type, in_chns=3, class_num=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    db_val = BaseDataSets(base_dir=root_path, fold=fold, split="val", transform=transforms.Compose([
        RandomGenerator(patch_size)
    ]))
    
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=32)
    
    total_dice_scores = [0, 0]
    total_HD95_scores = [0, 0]
    
    
    num_batches = 0
    for i_batch, sampled_batch in enumerate(valloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        
                    
        with torch.no_grad():
            if net_type == "unet_cct" or net_type == "unet_tri" or net_type == "unet_edge" or net_type == 'unet_tri_att':
                outputs = model(volume_batch)[0]
                # outputs = model(volume_batch)
            else:
                outputs = model(volume_batch)
            if len(outputs[0].shape) == 4 :
                outputs = outputs[0]    
            preds = torch.argmax(outputs, dim=1)
            
            # 计算指标
            dices, HD95 = calculate_metrics(label_batch, preds, num_classes)

                        
            for cls in range(num_classes - 1):
                total_dice_scores[cls] += dices[cls]
                total_HD95_scores[cls] += HD95[cls]

            num_batches += 1
                    
                
    # 计算平均指标
    avg_dice_scores = [total / num_batches for total in total_dice_scores]
    avg_HD95_scores = [total / num_batches for total in total_HD95_scores]

                
    # 记录指标
    avg_dice_scores = [score.cpu().numpy() for score in avg_dice_scores]
    avg_HD95_scores = [score for score in avg_HD95_scores]

            
    # logging.info(f'iteration {iter_num} :  Dice: {str(avg_dice_scores)},  IoU: {str(avg_iou_scores)},  Pixel Acc: {str(avg_pixel_accs)}')
    print(f'category 0:  Dice: {avg_dice_scores[0]:.3f},  HD95: {avg_HD95_scores[0]:.3f}')
    print(f'category 1:  Dice: {avg_dice_scores[1]:.3f},  HD95: {avg_HD95_scores[1]:.3f}')
    # print(f'category 2:  Dice: {avg_dice_scores[2]:.3f},  HD95: {avg_HD95_scores[2]:.3f}')
            
            
    avg_dice = np.mean(avg_dice_scores)
    avg_HD95 = np.mean(avg_HD95_scores)

        
    print(f'Total:       Dice: {avg_dice:.3f},  IoU: {avg_HD95:.3f}')
    
    
    df = pd.DataFrame({
    "Total_Dice": [avg_dice],
    "Total_HD95": [avg_HD95],
    "category_0_Dice": [avg_dice_scores[0]],
    "category_0_HD95": [avg_HD95_scores[0]],
    "category_1_Dice": [avg_dice_scores[1]],
    "category_1_HD95": [avg_HD95_scores[1]],
    # "category_2_Dice": [avg_dice_scores[2]],
    # "category_2_HD95": [avg_HD95[2]],
})
    
    df = df.astype(float).round(3)

    return df


def find_files(root_folder, file_suffix):
        matches = []
        for root, dirs, files in os.walk(root_folder):
            for filename in files:
                if filename.endswith(file_suffix):
                    matches.append(os.path.join(root, filename))
        return matches

if __name__ == "__main__":
    
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)
    

    # 设置基础路径和要查找的文件后缀
    results_path = '/home/cj/code/model/threshold/result'
    base_path = '/home/cj/code/model/threshold/threshold_cell'
    file_suffix = 'best_model.pth'
    root_path = '/home/cj/scribble_data/final_h5_ordered_edges/'
    patch_size = [256, 256]
    num_classes = 3
    
    # 调用函数进行搜索
    model_paths = find_files(base_path, file_suffix)
    model_paths.sort()
    
    final_df = pd.DataFrame()

    
    for model_path in tqdm(model_paths):
        parts = model_path.split(os.sep)
        net_type = parts[-1].split('_best_model.pth')[0]
        fold = parts[-3].split('_')[-1]
        model_name = parts[-3].split('_'+fold)[0]
        print(model_name, net_type, fold)
        df = val(root_path, fold, patch_size, num_classes, net_type, model_path)
        
        # 创建包含 model_name, net_type, fold 的 DataFrame
        model_info_df = pd.DataFrame({
            'model_name': [model_name],
            'net_type': [net_type],
            'fold': [fold]
        })

        # fileName = os.path.join(results_path , 'val_all_fold_results.csv')
        # book = load_workbook(fileName)
        # writer = pd.ExcelWriter(fileName, engine='openpyxl')
        # writer.book = book
        # writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        # 拼接模型信息和验证结果
        df_combined = pd.concat([model_info_df, df], axis=1)
        
        # 将结果累积到最终 DataFrame 中
        final_df = pd.concat([final_df, df_combined])
        df_combined.to_csv(results_path + '/val_all_fold_results.csv', mode='a', header=False, index=False)
        # writer.save()
        
    # # 重置索引
    final_df.reset_index(drop=True, inplace=True)
    
    # final_df.to_csv(results_path + 'val_all_fold_results.csv', index=False)
    
    # 选择 final_df 中的数值列
    numeric_df = final_df.select_dtypes(include=[np.number])
    
    # 在 numeric_df 中重新添加 'model_name' 列
    numeric_df_with_name = final_df[['model_name']].join(numeric_df)

    # 分组并计算每个模型名称的平均指标
    avg_results = numeric_df_with_name.groupby('model_name').mean()

    # 打印结果或将结果保存到CSV文件
    print(avg_results)
    avg_results = avg_results.astype(float).round(3)
    
    avg_results.to_csv(results_path + 'avg_results_by_model_' + base_path.split(os.sep)[-2] + '.csv', index=True)


        
    
