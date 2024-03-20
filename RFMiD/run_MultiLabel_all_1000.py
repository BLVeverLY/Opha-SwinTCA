import os
import argparse
# import ruamel.yaml as yaml
from collections import Counter
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import _utils, ConcatDataset # import torch.utils.data
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    roc_auc_score, f1_score, precision_recall_fscore_support, roc_curve, auc, classification_report
from sklearn.metrics import multilabel_confusion_matrix, zero_one_loss, precision_score, recall_score
from matplotlib import pyplot as plt
from itertools import cycle
import torchvision
from torchvision import transforms

from RFMiD import MultiLabel, MultiLabelMultiModality
from model_swintransformer import MyModel_swin_cat, MyModel_swin_trans, MyModel_swin_6channel
from model_one import MyModel_res_1, MyModel_swin_1, MyModel_res_2, MyModel_swin_2

TRAIN_EPOCH_START = 0
EPOCH_TRAIN = 1000
EPOCH_TEST = 1
NUM_CLASSES = 45
LEARNING_RATE = 1e-5

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


def train(args, model, sampler, train_dataloader, val_dataloader):
    print("Start training.")
    if args.checkpoint_load:
        model.load_state_dict(torch.load(args.checkpoint_load), strict=False) # torch.load(path, map_location='cuda:0')
        TRAIN_EPOCH_START = 0
        print("Training from ", TRAIN_EPOCH_START, ", loading parameters from ", args.checkpoint_load)
    else:
        TRAIN_EPOCH_START = 0

    model.train()
    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda() # gpu 0/1/2
    criterion = nn.BCEWithLogitsLoss().cuda() # torch.sigmoid+torch.nn.BCELoss == torch.nn.BCEWithLogitsLoss
    # https://blog.csdn.net/qsmx666/article/details/121718548
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    
    loss_list = list()
    train_accuracy_list1 = list()
    train_accuracy_list2 = list()
    val_accuracy_list1 = list()
    val_accuracy_list2 = list()
    for epoch in range(TRAIN_EPOCH_START, EPOCH_TRAIN): # args.epochs
        sampler.set_epoch(epoch) # 每次epoch打乱顺序
        total_loss = 0 # 单个GUP上每个epoch的总loss
        total_num = 0 # 单个GUP上每个epoch的总num
        # total_right = 0 # 每个epoch在训练集上的总right个数
        train_acc1 = [] # 单个GUP上的数据在训练集上的accuracy list
        train_acc2 = []
        # train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
        #                 loss_scaler)
        # for i, (image1, image2, label, _) in enumerate(dataloader):
        batch = tqdm(train_dataloader)
        batch.set_description(f'Epoch [{epoch+1}/{EPOCH_TRAIN}] RANK [{args.gpu}]')
        for image, label in batch:
            images = image.cuda(non_blocking=True) # 会加载到args.gpu对应的GPU上
            label = label.cuda(non_blocking=True) # torch.Size([batch_size])
            # Forward pass
            outputs = model(images) # torch.Size([batch_size, num_classes])
            loss = criterion(outputs, label.float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
            total_num += len(label)
            batch.set_postfix(loss=total_loss/total_num, acc1=accuracy_score(label.cpu().numpy(), torch.sigmoid(outputs).ge(0.5).cpu().numpy()),acc2=zero_one_loss(label.cpu().numpy(),torch.sigmoid(outputs).ge(0.5).cpu().numpy()))
            # right = (outputs.argmax(1)==labels).sum()
            # total_right = total_right + right ！！！最后除以数量，问题是样本数量不等于train_size
            # 在multilabel（多标签问题）分类中，accuracy_score函数会返回子集的准确率。如果对于一个样本来说，必须严格匹配真实数据集中的label，整个集合的预测标签返回1.0；否则返回0.0.
            train_acc1.append(accuracy_score(label.cpu().numpy(), torch.sigmoid(outputs).ge(0.5).cpu().numpy()))
            # 0-1损失计算的是完全预测错误的样本占总样本的比例。
            train_acc2.append(zero_one_loss(label.cpu().numpy(),torch.sigmoid(outputs).ge(0.5).cpu().numpy()))

        if((epoch+1)%20 == 0): # 每10个epoch输出/记录一次loss和accuracy
            model.eval()
            val_acc1 = []
            val_acc2 = []
            with torch.no_grad():
                for i, (image, label) in enumerate(val_dataloader):
                    val_outputs = model(image.cuda(non_blocking=True))
                    # val_acc1.append(accuracy_score(label.numpy(), torch.sigmoid(val_outputs).ge(0.5).cpu().numpy()))
                    # val_acc2.append(zero_one_loss(label.cpu().numpy(),torch.sigmoid(outputs).ge(0.5).cpu().numpy()))
                # print("Epoch: ", epoch+1, "; Loss: ", total_loss/total_num, "; Train Acuracy: ", np.mean(np.array(train_acc1)),np.mean(np.array(train_acc2)), "Val Accuracy:", np.mean(np.array(val_acc1)),np.mean(np.array(val_acc2)))
                loss_list.append(total_loss/total_num) # 每个sample上的loss
                # train_accuracy_list1.append(np.mean(np.array(train_acc1)))
                # train_accuracy_list2.append(np.mean(np.array(train_acc2)))
                # val_accuracy_list1.append(np.mean(np.array(val_acc1)))
                # val_accuracy_list2.append(np.mean(np.array(val_acc2)))
            model.train()

        if((epoch+1)%500 == 0):  # 每500个epoch存一次参数
            torch.save(model.state_dict(), "./save/"+args.model+"_"+str(epoch+1)+".pth")

    if args.checkpoint_save:
        torch.save(model.state_dict(), "./save/"+args.checkpoint_save)
    else:
        print("Automatically save the checkpoint into './save/MODEL_EPOCH_TRAIN.pth'")
        torch.save(model.state_dict(), "./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth")
    np.save("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_loss.npy", torch.tensor(loss_list, device = 'cpu'))
    # np.save("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_train_accuracy1.npy", torch.tensor(train_accuracy_list1, device = 'cpu'))
    # np.save("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_train_accuracy2.npy", torch.tensor(train_accuracy_list2, device = 'cpu'))
    # np.save("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_val_accuracy1.npy", torch.tensor(val_accuracy_list1, device = 'cpu'))
    # np.save("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_val_accuracy2.npy", torch.tensor(val_accuracy_list2, device = 'cpu'))
    plot(args, "loss")
    # plot(args, "train_accuracy1")
    # plot(args, "train_accuracy2")
    # plot(args, "val_accuracy1")
    # plot(args, "val_accuracy2")
    return

label_onehot_array = np.zeros(1, dtype=int)
score_array = np.zeros(1, dtype=int)

def eval(args, model, dataloader):
    score_list = []
    label_list = []
    global label_onehot_array
    global score_array

    print("Start evaluating.")
    if args.checkpoint_load:
        model.load_state_dict(torch.load(args.checkpoint_load)) # torch.load(path, map_location='cuda:0')
    else:
        if os.path.isfile("./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth"):
            print("Automatically load the checkpoint from './save/MODEL_EPOCH_TRAIN.pth'")
            model.load_state_dict(torch.load("./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth")) # torch.load(path, map_location='cuda:0')
        else:
            print("Without using parameters.") # If trained, the program will automatically using training parameters.
        
    model.eval()

    total_accuracy = 0
    y_label = []
    y_pred = []

    with torch.no_grad():
        # for i, (image1, image2, label, person_id) in enumerate(dataloader):
        for image, label in tqdm(dataloader):
            images = image.cuda(non_blocking=True)
            labels = label.cuda(non_blocking=True) 
            outputs = model(images)
            outputs = torch.sigmoid(outputs).cpu()
            # accuracy = (outputs.argmax(1)==labels).sum()
            # accuracy = sum(row.all().int().item() for row in (outputs.ge(0.5) == labels))
            total_accuracy = total_accuracy + accuracy
            y_label = y_label + label.tolist()
            y_pred = y_pred + outputs.ge(0.5).tolist()
            # y_pred = y_pred + np.round(outputs).tolist()

            output = outputs.cpu().numpy()
            target = label.cpu().numpy()
            # confusion.update(output_1[0], target_1[0])
    
            score_list.extend(output)
            label_list.extend(target) 
    
    if score_array.size == 1:
        score_array = np.array(score_list)
    else:
        score_array = np.concatenate((score_array, np.array(score_list)))
    # # label to onehot
    # label_tensor = torch.tensor(label_list)
    # label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    # label_onehot = torch.zeros(label_tensor.shape[0], NUM_CLASSES)
    # label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    # if label_onehot_array.size == 1:
    #     label_onehot_array = np.array(label_onehot)
    # else:
    #     label_onehot_array = np.concatenate((label_onehot_array, np.array(label_onehot)))
    print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
    print("label_onehot_array:", label_onehot_array.shape)  # torch.Size([batchsize, classnum]) onehot

    return total_accuracy, y_label, y_pred

def Hamming_Loss(y_true, y_pred):
    count = 0
    for i in range(len(y_true)):
        # 单个样本的标签数
        p = np.size(y_true[i] == y_pred[i])
        # np.count_nonzero用于统计数组中非零元素的个数
        # 单个样本中预测正确的样本数
        q = np.count_nonzero(y_true[i] == y_pred[i])
        print(f"{p}-->{q}")
        count += p - q
    print(f"样本数：{len(y_true)}, 标签数：{NUM_CLASSES}") # 样本数：3, 标签数：4
    return count / (len(y_true) * NUM_CLASSES)


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

    # dataset = TwoImageData()
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    if args.model == 'swin_trans_RFMiD':
        train_dataset = MultiLabelMultiModality(data_folder="/home/newljy/RFMiD_All_Classes_Dataset/1. Original Images/a. Training Set",
                        label_path="/home/newljy/RFMiD_All_Classes_Dataset/2. Groundtruths/a. RFMiD_Training_Labels.csv")
        test_dataset = MultiLabelMultiModality(data_folder="/home/newljy/RFMiD_All_Classes_Dataset/1. Original Images/c. Testing Set",
                        label_path="/home/newljy/RFMiD_All_Classes_Dataset/2. Groundtruths/c. RFMiD_Testing_Labels.csv")
        val_dataset = MultiLabelMultiModality(data_folder="/home/newljy/RFMiD_All_Classes_Dataset/1. Original Images/b. Validation Set",
                        label_path="/home/newljy/RFMiD_All_Classes_Dataset/2. Groundtruths/b. RFMiD_Validation_Labels.csv")
        print(train_dataset.__len__())
        print(test_dataset.__len__())
        print(val_dataset.__len__())
    elif args.model == 'swin_one_RFMiD':
        train_dataset = MultiLabel(data_folder="/home/newljy/RFMiD_All_Classes_Dataset/1. Original Images/a. Training Set",
                        label_path="/home/newljy/RFMiD_All_Classes_Dataset/2. Groundtruths/a. RFMiD_Training_Labels.csv")
        test_dataset = MultiLabel(data_folder="/home/newljy/RFMiD_All_Classes_Dataset/1. Original Images/c. Testing Set",
                        label_path="/home/newljy/RFMiD_All_Classes_Dataset/2. Groundtruths/c. RFMiD_Testing_Labels.csv")
        val_dataset = MultiLabel(data_folder="/home/newljy/RFMiD_All_Classes_Dataset/1. Original Images/b. Validation Set",
                        label_path="/home/newljy/RFMiD_All_Classes_Dataset/2. Groundtruths/b. RFMiD_Validation_Labels.csv")
        print(train_dataset.__len__())
        print(test_dataset.__len__())
        print(val_dataset.__len__())
    # train_dataset = ConcatDataset([train_dataset, test_dataset])

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset) # if is_distributed else None
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_sampler = DistributedSampler(test_dataset) # if is_distributed else None
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=test_sampler)
        val_sampler = DistributedSampler(val_dataset) # if is_distributed else None
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, sampler=test_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size)


    # loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
    # for epoch in range(start_epoch, n_epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)
    #         # 在分布式模式下，需要在每个 epoch 开始时调用 set_epoch() 方法，然后再创建 DataLoader 迭代器，
    #         # 以使 shuffle 操作能够在多个 epoch 中normal工作。 否则，dataloader迭代器产生的数据将始终使用相同的顺序。
    #         train(train_dataloader)

    if args.model == 'swin_trans_RFMiD':
        model = MyModel_swin_trans(NUM_CLASSES)
    elif args.model == 'swin_one_RFMiD':
        model = MyModel_swin_2(NUM_CLASSES)
    else: 
        print("Model does not exit.")
        exit
    model = model.to(device)
    # if torch.cuda.device_count()>1: # Wrap the model
    model = nn.parallel.DistributedDataParallel(model) # gpu 0,1,2

    if args.eval == 'False': # train
        train(args, model, train_sampler, train_dataloader, val_dataloader)

    y_label_list = []
    y_pred_list = []
    nagetive_tulpe_list = []
    accuracy = 0
    for epoch in range(EPOCH_TEST):
        print('Eval: ', epoch+1)
        test_sampler.set_epoch(epoch) # 每次epoch打乱顺序
        total_accuracy, y_label, y_pred = eval(args, model, test_dataloader)

        dist.barrier()  # synchronizes all processes
        # dist.reduce(total_accuracy, op=torch.distributed.ReduceOp.SUM, dst=0)
        total_accuracy = y_label_all_gpu = y_pred_all_gpu = [None for _ in range(torch.cuda.device_count())] # 申明
        # dist.all_gather_object(nagetive_person_id_all_gpu, nagetive_person_id)
        # nagetive_person_id_all_gpu = [i for ii in nagetive_person_id_all_gpu for i in ii] # 降维
        # dist.all_gather_object(nagetive_y_label_all_gpu, nagetive_y_label)
        # nagetive_y_label_all_gpu = [i for ii in nagetive_y_label_all_gpu for i in ii] # 降维
        # dist.all_gather_object(nagetive_y_pred_all_gpu,  nagetive_y_pred)
        # nagetive_y_pred_all_gpu = [i for ii in  nagetive_y_pred_all_gpu for i in ii] # 降维
        # dist.gather_object(y_label_all_gpu, y_label, dst=0) # 哈哈，会报错
        dist.all_gather_object(y_label_all_gpu, y_label)
        y_label_all_gpu = [i for ii in y_label_all_gpu for i in ii] # 降维
        dist.all_gather_object(y_pred_all_gpu, y_pred)
        y_pred_all_gpu = [i for ii in y_pred_all_gpu for i in ii] # 降维

        # all_number_one_gpu = int(len(test_dataset)/torch.cuda.device_count() + \
        #                          (args.batch_size - len(test_dataset)/torch.cuda.device_count())%args.batch_size)
        # y_label_all_gpu = [torch.zeros(all_number_one_gpu, dtype=torch.int32) for _ in range(torch.cuda.device_count())]
        # y_pred_all_gpu = [torch.zeros(all_number_one_gpu, dtype=torch.int32) for _ in range(torch.cuda.device_count())]
        # # y_label = torch.IntTensor(y_label).cuda(args.gpu)
        # y_label = torch.IntTensor(y_label).cuda()
        # dist.gather(y_label_all_gpu, y_label, dst=0) # 哈哈，会报错

        # nagetive_tulpe_all_gpu = list(zip(nagetive_person_id_all_gpu, nagetive_y_label_all_gpu, nagetive_y_pred_all_gpu))
        # nagetive_tulpe_list = nagetive_tulpe_list + nagetive_tulpe_all_gpu
        y_label_list = y_label_list + y_label_all_gpu
        y_pred_list = y_pred_list + y_pred_all_gpu
        # accuracy = accuracy + total_accuracy
    # nagetive = Counter(nagetive_tulpe_list)
    # d_nagetive = sorted(nagetive.items(), key=lambda x: x[1], reverse=True)
    # print("(zip(nagetive_person_id, nagetive_y_label, nagetive_y_pred), wrong_predict_times)")
    # print(nagetive)
    # print("In decreasing order based on wrong_predict_times:")
    # print(d_nagetive)

    # print(f"Total accuracy: {accuracy/(len(test_dataset)*EPOCH_TEST)}")

    # plot(args, "loss")
    # plot(args, "train accuracy")
    # plot(args, "val accuracy")
    # 必须放在CM前面，哈哈

    # print('y_label_list and y_pred_list:')
    # print(y_label_list)
    # print(y_pred_list)

    # plt.cla()
    # # confusion_mat = confusion_matrix(y_label_list, y_pred_list)
    # confusion_mat = multilabel_confusion_matrix(y_label_list, y_pred_list)
    # print(confusion_mat)
    # classes = ['DR','ARMD','MH','DN','MYA','BRVO','TSLN','ERM','LS','MS','CSR','ODC','CRVO','TV','AH','ODP','ODE','ST','AION','PT','R','RS','CRS','EDN','RPEC','MHL','RP','CWS','CB','ODPM','PRH','MNF','HR','CRAO','TD','CME','PTCR','CF','VH','MCA','VS','BRAO','PLQ','HPED','CL']
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    # # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat) # 傻逼玩意儿版本不对，自定义标签值能生成图片但会报错
    # disp.plot(
    #     include_values=True,
    #     # cmap="viridis",
    #     cmap="Greens",
    #     ax=None,
    #     xticks_rotation="horizontal",
    #     values_format="d"     
    # )
    # plt.savefig("./save/CM_"+args.model+"_"+str(EPOCH_TRAIN)+".png")
    
    # plt.cla()
    # confusion_mat = confusion_matrix(y_label_list, y_pred_list, normalize='true')
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    # # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    # disp.plot(
    #     include_values=True,
    #     # cmap="viridis",
    #     cmap="Greens",
    #     ax=None,
    #     xticks_rotation="horizontal",
    #     # values_format="f"
    # )
    # plt.savefig("./save/CM_"+args.model+"_"+str(EPOCH_TRAIN)+"_row_normalize.png")

    print("1", accuracy_score(y_label_list,y_pred_list))
    print("2", zero_one_loss(y_label_list,y_pred_list))
    print("3", precision_score(y_true=y_label_list, y_pred=y_pred_list, average='samples',zero_division=1))
    print("4", recall_score(y_true=y_label_list, y_pred=y_pred_list, average='samples',zero_division=1))
    print("5", f1_score(y_label_list,y_pred_list,average='samples',zero_division=1))
    print("6", Hamming_Loss(y_label_list,y_pred_list))

# print(classification_report(y_label_list, y_pred_list, digits=4))

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
# plt.savefig("./ROC.png")

# print("micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]))
# print("macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]))