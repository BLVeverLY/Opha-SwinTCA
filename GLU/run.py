import os
import argparse
# import ruamel.yaml as yaml
from collections import Counter
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import _utils, ConcatDataset, SubsetRandomSampler # import torch.utils.data
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    roc_auc_score, f1_score, precision_recall_fscore_support, roc_curve, auc, classification_report
from matplotlib import pyplot as plt
from itertools import cycle
from sklearn.model_selection import KFold
from scipy import stats

from data_process import TwoImageData
# from data_process_new import TwoImageData
from model_resnet import MyModel_res_cat, MyModel_res_trans, MyModel_res_6channel
from model_swintransformer import MyModel_swin_cat, MyModel_swin_trans, MyModel_swin_6channel
from model_one import MyModel_res_1, MyModel_swin_1, MyModel_res_2, MyModel_swin_2

TRAIN_EPOCH_START = 0
EPOCH_TRAIN = 100
EPOCH_TEST = 1
NUM_CLASSES = 2
LEARNING_RATE = 1e-4

def plot(args, name):
    plt.cla()
    y = list(np.load("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_"+name+".npy"))
    x = list(range(20,EPOCH_TRAIN+1,20))
    plt.xlim(0,EPOCH_TRAIN)
    # plt.ylim(0,1)
    plt.plot(x, y, '.-')
    plt.autoscale(True)
    plt_title = 'BATCH_SIZE = '+str(args.batch_size)+' ; LEARNING_RATE = '+str(LEARNING_RATE)
    plt.title(plt_title)
    plt.xlabel('per 20 epochs')
    plt.ylabel(name)
    plt.savefig("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_"+name+".png")


# def train(args, model, sampler, train_dataloader, val_dataloader):
def train(args, model, train_dataloader):
    print("Start training.")
    if args.checkpoint_load:
        dict = torch.load(args.checkpoint_load, map_location='cuda:0')
        dict_new = {}
        for k, v in dict.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            dict_new[new_k] = v
        model.load_state_dict(dict_new, strict=False) # torch.load(path, map_location='cuda:0')
        TRAIN_EPOCH_START = 0
        print("Training from ", TRAIN_EPOCH_START, ", loading parameters from ", args.checkpoint_load)
        for param in model.named_parameters():
            if param[0] in dict:
                print(param[0])
                param[1].requires_grad = False
        model = model.to(device)
        # if torch.cuda.device_count()>1: # Wrap the model
        model = nn.parallel.DistributedDataParallel(model) # gpu 0,1,2
    else:
        TRAIN_EPOCH_START = 0

    model.train()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda() # gpu 0/1/2
    # optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    ## optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE,momentum=args.momentum, weight_decay=args.weight_decay)
    
    loss_list = list()
    train_accuracy_list = list()
    val_accuracy_list = list()
    for epoch in range(TRAIN_EPOCH_START, EPOCH_TRAIN): # args.epochs
        # sampler.set_epoch(epoch) # 每次epoch打乱顺序
        total_loss = 0 # 单个GUP上每个epoch的总loss
        total_num = 0 # 单个GUP上每个epoch的总num
        # total_right = 0 # 每个epoch在训练集上的总right个数
        train_acc = [] # 单个GUP上的数据在训练集上的accuracy list
        # train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
        #                 loss_scaler)
        # for i, (image1, image2, label, _) in enumerate(dataloader):
        batch = tqdm(train_dataloader)
        batch.set_description(f'Epoch [{epoch+1}/{EPOCH_TRAIN}] RANK [{args.gpu}]')
        for image1, image2, label, _ in batch:
            images1 = image1.cuda(non_blocking=True) # 会加载到args.gpu对应的GPU上
            images2 = image2.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True) # torch.Size([batch_size])
            # Forward pass
            outputs = model(images1, images2) # torch.Size([batch_size, num_classes])
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
            total_num += len(label)
            batch.set_postfix(loss=total_loss/total_num)
            # right = (outputs.argmax(1)==labels).sum()
            # total_right = total_right + right ！！！最后除以数量，问题是样本数量不等于train_size
            train_acc.append(accuracy_score(label.cpu().numpy(), torch.max(outputs, 1)[1].cpu().numpy()))

        if((epoch+1)%500 == 0):  # 每500个epoch存一次参数
            torch.save(model.state_dict(), "./save/"+args.model+"_"+str(epoch+1)+".pth")

    if args.checkpoint_save:
        torch.save(model.state_dict(), "./save/"+args.checkpoint_save)
    else:
        print("Automatically save the checkpoint into './save/MODEL_EPOCH_TRAIN.pth'")
        torch.save(model.state_dict(), "./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth")
    return


def eval(args, model, dataloader):
    score_list = []
    label_list = []
    label_onehot_array = np.zeros(1, dtype=int)
    score_array = np.zeros(1, dtype=int)

    print("Start evaluating.")
    # if args.checkpoint_load:
    #     model = model.to(device)
    #     model = nn.parallel.DistributedDataParallel(model) # gpu 0,1,2
    #     model.load_state_dict(torch.load(args.checkpoint_load)) # torch.load(path, map_location='cuda:0')
    # else:
    model = model.to(device)
    # if torch.cuda.device_count()>1: # Wrap the model
    model = nn.parallel.DistributedDataParallel(model) # gpu 0,1,2
    if os.path.isfile("./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth"):
        print("Automatically load the checkpoint from './save/MODEL_EPOCH_TRAIN.pth'")
        model.load_state_dict(torch.load("./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth")) # torch.load(path, map_location='cuda:0')
    # else:
    #     print("Without using parameters.") # If trained, the program will automatically using training parameters.
    
    model.eval()
    # params = list(model.named_parameters())
    # print(params.__len__())
    # print(params[0])
    criterion = nn.CrossEntropyLoss().cuda() # gpu 0/1/2

    correct = 0
    total = 0
    y_true = []
    y_score = []
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    with torch.no_grad():
        # for i, (image1, image2, label, person_id) in enumerate(dataloader):
        for image1, image2, label, person_id in tqdm(dataloader):
            
            images1 = image1.cuda(non_blocking=True)
            images2 = image2.cuda(non_blocking=True)
            labels = label.cuda(non_blocking=True) 
            outputs = model(images1, images2) # torch.Size([batch_size, num_classes])
            
            total += label.size(0)
            correct += (outputs.argmax(1)==labels).sum().item()
            
            output = outputs.cpu().numpy()
            target = label.cpu().numpy()
            score_list.extend(output)
            label_list.extend(target)
                
        
        accuracy = correct / total

        score_array = np.array(score_list)
        # label to onehot
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], NUM_CLASSES)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        if label_onehot_array.size == 1:
            label_onehot_array = np.array(label_onehot)
        else:
            label_onehot_array = np.concatenate((label_onehot_array, np.array(label_onehot)))
        # print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
        # print("label_onehot_array:", label_onehot_array.shape)  # torch.Size([batchsize, classnum]) onehot

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(label_onehot_array[:, i], score_array[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(NUM_CLASSES):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= NUM_CLASSES
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return accuracy, roc_auc["macro"]


def setup_for_distributed(is_master):
    # This function disables printing when not in master process
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    print('Distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', help='distributed training backend')
    parser.add_argument('--model', type=str, required=True, help="swin or resnet, cat or trans")
    parser.add_argument('--batch_size', type=int, default=80,help="batch size for each gpu during training or evaluating;"
                                                        "the batch size in total equals batch_size*NPROC_PER_NODE*NNODES")
    parser.add_argument('--checkpoint_save', default='', type=str, help="local path for parameters.") # after train
    parser.add_argument('--eval', default=False, type=str, help="evaluation only")
    parser.add_argument('--checkpoint_load', default='', type=str, help="local path for parameters") # before eval
    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    init_distributed_mode(args)
    device = torch.device(args.device) # 指定使用所有GPU, 所有GPU上

    ## output config informations
    # yaml.dump(config, open('config.yaml', 'w'))
    # shutil.copy('config.yaml', config.output_dir) # shutil.copy

    # train_dataset = test_dataset = TwoImageData()
    # dataset = TwoImageData()
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset = TwoImageData(isTrain=True)
    test_dataset = TwoImageData(isTrain=False)
    train_dataset = ConcatDataset([train_dataset, test_dataset])

    # if args.distributed:
    #     train_sampler = DistributedSampler(train_dataset) # if is_distributed else None
    #     train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    #     test_sampler = DistributedSampler(test_dataset) # if is_distributed else None
    #     test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=test_sampler)
    # else:
    #     train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    #     test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)


    # loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
    # for epoch in range(start_epoch, n_epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)
    #         # 在分布式模式下，需要在每个 epoch 开始时调用 set_epoch() 方法，然后再创建 DataLoader 迭代器，
    #         # 以使 shuffle 操作能够在多个 epoch 中normal工作。 否则，dataloader迭代器产生的数据将始终使用相同的顺序。
    #         train(train_dataloader)

    if args.model == 'swin_trans':
        model = MyModel_swin_trans(NUM_CLASSES)
    else: 
        print("Model does not exit.")
        exit
    # model = model.to(device)
    # # if torch.cuda.device_count()>1: # Wrap the model
    # model = nn.parallel.DistributedDataParallel(model) # gpu 0,1,2


    if args.eval == 'False': # train
        accuracies = []
        auc_scores = []
        # 创建k-fold交叉验证对象
        num_folds = 5
        kfold = KFold(n_splits=num_folds, shuffle=True)
        # 进行k-fold交叉验证
        fold = 1
        for train_indices, test_indices in kfold.split(train_dataset):
            print('Fold {}/{}'.format(fold, num_folds))
            # 创建数据加载器
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=test_sampler)
            # train_loader = torch.utils.data.DataLoader(dataset=train_dataset[train_indices], batch_size=args.batch_size, shuffle=True)
            # test_loader = torch.utils.data.DataLoader(dataset=train_dataset[test_indices], batch_size=args.batch_size, shuffle=False)
            
            train(args, model, train_loader)
            
            accuracy, auc_score = eval(args, model, test_loader)
            accuracies.append(accuracy)
            auc_scores.append(auc_score)
            
            # 更新fold计数器
            fold += 1

        # 计算平均准确率
        avg_accuracy = np.mean(accuracies)
        print(accuracies)

        # 计算置信区间
        confidence_interval = stats.t.interval(0.95, len(accuracies)-1, loc=avg_accuracy, scale=stats.sem(accuracies))

        print("Average Accuracy: {:.4f} ".format(avg_accuracy))
        print("Confidence Interval: ({:.4f} , {:.4f} )".format(confidence_interval[0], confidence_interval[1]))
        
        # 计算平均macro AUC
        avg_auc_score = np.mean(auc_scores)
        print(auc_scores)

        # 计算置信区间
        confidence_interval2 = stats.t.interval(0.95, len(auc_scores)-1, loc=avg_auc_score, scale=stats.sem(auc_scores))

        print("Average Accuracy: {:.4f} ".format(avg_auc_score))
        print("Confidence Interval: ({:.4f} , {:.4f} )".format(confidence_interval2[0], confidence_interval2[1]))

        #axes[0]表示在第一张图的轴上描点画图
        #vert=True表示boxplot图是竖着放的
        #patch_artist=True 表示填充颜色
        plt.boxplot([accuracies, auc_scores],
                    vert=True, 
                    patch_artist=True)
        plt.ylim(0, 1)
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        plt.xticks([1,2],['accuracy','macro AUC score'])
        plt.savefig('boxplot.png')



# f1_micro = f1_score(y_label_list, y_pred_list,average='micro')
# f1_macro = f1_score(y_label_list, y_pred_list,average='macro')
# print(f1_micro, f1_macro) # 选macro

# p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_label_list, y_pred_list,labels=[0,1,2,3,4,5]) 
# print(p_class)
# print(r_class)
# print(f_class)
# print(support_micro)

# # sklearn
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(NUM_CLASSES):
#     fpr[i], tpr[i], _ = roc_curve(label_onehot_array[:, i], score_array[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(label_onehot_array.ravel(), score_array.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))

# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(NUM_CLASSES):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
# mean_tpr /= NUM_CLASSES

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plot all ROC curves
# plt.figure()
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]),
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label="macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]),
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )

# colors = cycle(["aqua", "darkorange", "cornflowerblue"])
# lw = 2
# for i, color in zip(range(NUM_CLASSES), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=lw,
#         label="ROC curve of class {0} (area = {1:0.4f})".format(i, roc_auc[i]),
#     )

# plt.plot([0, 1], [0, 1], "k--", lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Some extension of Receiver operating characteristic to multiclass")
# plt.legend(loc="lower right")
# plt.savefig("./save/ROC_"+args.model+"_"+str(EPOCH_TRAIN)+".png")

# print("micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]))
# print("macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]))