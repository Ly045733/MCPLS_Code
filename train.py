import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss, BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.dataset_edge import EdgeBaseDataSets, EdgeRandomGenerator
from dataloaders.dataset import BaseDataSets, RandomGenerator   
from networks.net_factory import net_factory
from my_utils import losses


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/SegPC_2021_Edge', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='cell_pCE_SPS', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet_tri', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=3000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')

parser.add_argument('--edge_paras', type=str, default="30_40_0",help='edge_paras')
args = parser.parse_args()

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def recall_score(y_true, y_pred, smooth=1e-6):
    # 计算真正例（TP）和假负例（FN）
    TP = torch.sum((y_true == 1) & (y_pred == 1))
    FN = torch.sum((y_true == 1) & (y_pred == 0))
    return (TP + smooth) / (TP + FN + smooth)  # 添加小的常数避免除以零

def calculate_metrics(y_true, y_pred, num_classes):
    dices = []
    ious = []
    accuracies = []
    recalls = []

    for cls in range(num_classes):
        cls_y_true = (y_true == cls).long()
        cls_y_pred = (y_pred == cls).long()

        dices.append(dice_coefficient(cls_y_true, cls_y_pred))
        ious.append(iou_score(cls_y_true, cls_y_pred))
        accuracies.append(torch.mean((cls_y_true == cls_y_pred).float()))
        recalls.append(recall_score(cls_y_true, cls_y_pred))

    return dices, ious, accuracies, recalls

def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length



def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
    
    db_train = EdgeBaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        EdgeRandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type, edge_paras="30_40_0")
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=32, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=32)
    
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=3)
    bce_loss = BCELoss() 
    mse_loss = MSELoss()
    dice_loss = losses.pDLoss(num_classes, ignore_index=3)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    
    best_dice = 0.0
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, edge_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['edge']
            volume_batch, label_batch, edge_batch = volume_batch.cuda(), label_batch.cuda(), edge_batch.cuda()
        
            outputs, outputs_aux1, outputs_aux2 = model(
                volume_batch)
            outputs_soft1 = torch.softmax(outputs, dim=1)
            outputs_soft2 = torch.softmax(outputs_aux1, dim=1)
            
            # print(outputs.shape, outputs_aux1.shape, outputs_soft1.shape, outputs_soft2.shape, volume_batch.shape, label_batch.shape)

            loss_ce1 = ce_loss(outputs, label_batch[:].long())
            loss_ce2 = ce_loss(outputs_aux1, label_batch[:].long())
            
            outputs_aux2 = outputs_aux2.squeeze(1)
            outputs_aux2_sigmoid = torch.sigmoid(outputs_aux2)
            loss_edge = bce_loss(outputs_aux2_sigmoid, edge_batch[:].float())
            
            
            loss_ce = 0.5 * (loss_ce1 + loss_ce2) + 0.1*loss_edge

            beta = random.random() + 1e-10

            pseudo_supervision = torch.argmax(
                (beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()), dim=1, keepdim=False)

            loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(
                1)) + dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1)))

            loss = loss_ce + 0.5 * loss_pse_sup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f' %
                (iter_num, loss.item(), loss_ce.item()))
            
            if iter_num % 20 == 0:
                image = volume_batch[1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
                
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                total_dice_scores = [0, 0, 0]
                total_iou_scores = [0, 0, 0]
                total_pixel_accs = [0, 0, 0]
                total_pixel_recalls = [0, 0, 0]
                num_batches = 0
                for i_batch, sampled_batch in enumerate(valloader):
                    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    
                    with torch.no_grad():
                        outputs = model(volume_batch)[0]
                        preds = torch.argmax(outputs, dim=1)
                        
                        # 计算指标
                        dices, ious, accuracies, recalls = calculate_metrics(label_batch, preds, num_classes)

                        
                        for cls in range(num_classes):
                            total_dice_scores[cls] += dices[cls]
                            total_iou_scores[cls] += ious[cls]
                            total_pixel_accs[cls] += accuracies[cls]
                            total_pixel_recalls[cls] += recalls[cls]

                    num_batches += 1
                    
                
                # 计算平均指标
                avg_dice_scores = [total / num_batches for total in total_dice_scores]
                avg_iou_scores = [total / num_batches for total in total_iou_scores]
                avg_pixel_accs = [total / num_batches for total in total_pixel_accs]
                avg_pixel_recalls = [total / num_batches for total in total_pixel_recalls]
                
                    
                # 记录指标
                avg_dice_scores = [score.cpu().numpy() for score in avg_dice_scores]
                avg_iou_scores = [score.cpu().numpy() for score in avg_iou_scores]
                avg_pixel_accs = [acc.cpu().numpy() for acc in avg_pixel_accs]
                avg_pixel_recalls = [recall.cpu().numpy() for recall in avg_pixel_recalls]
                
                # logging.info(f'iteration {iter_num} : Val Dice: {str(avg_dice_scores)}, Val IoU: {str(avg_iou_scores)}, Val Pixel Acc: {str(avg_pixel_accs)}')
                logging.info(f'iteration {iter_num} : category 0: Val Dice: {avg_dice_scores[0]:.4f}, Val IoU: {avg_iou_scores[0]:.4f}, Val Pixel Acc: {avg_pixel_accs[0]:.4f}, Val Recall: {avg_pixel_recalls[0]:.4f}')
                logging.info(f'iteration {iter_num} : category 1: Val Dice: {avg_dice_scores[1]:.4f}, Val IoU: {avg_iou_scores[1]:.4f}, Val Pixel Acc: {avg_pixel_accs[1]:.4f}, Val Recall: {avg_pixel_recalls[1]:.4f}')
                logging.info(f'iteration {iter_num} : category 2: Val Dice: {avg_dice_scores[2]:.4f}, Val IoU: {avg_iou_scores[2]:.4f}, Val Pixel Acc: {avg_pixel_accs[2]:.4f}, Val Recall: {avg_pixel_recalls[2]:.4f}')
                
                
                avg_dice = np.mean(avg_dice_scores)
                avg_iou = np.mean(avg_iou_scores)
                avg_pixel_acc = np.mean(avg_pixel_accs)
                writer.add_scalar('val/dice', avg_dice, iter_num)
                writer.add_scalar('val/iou', avg_iou, iter_num)
                writer.add_scalar('val/pixel_acc', avg_pixel_acc, iter_num)
                
                if avg_dice > best_dice:
                    best_performance = avg_dice
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    
                logging.info(f'iteration {iter_num} : Val Dice: {avg_dice}, Val IoU: {avg_iou}, Val Pixel Acc: {avg_pixel_acc}')
                    
                model.train()
                
            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
                
    writer.close()
    return "Training Finished!"
            


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)