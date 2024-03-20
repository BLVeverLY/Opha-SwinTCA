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
from matplotlib import pyplot as plt
from itertools import cycle
import torchvision
from torchvision import transforms

from model import MyModel_swin_OCT, MyModel_swin_fundus, MyModel_res_OCT, MyModel_res_fundus

TRAIN_EPOCH_START = 0
EPOCH_TRAIN = 1000
EPOCH_TEST = 1
NUM_CLASSES = 6
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


def train(args, model, sampler, train_dataloader, val_dataloader):
    print("Start training.")
    if args.checkpoint_load:
        model.load_state_dict(torch.load("./save/"+args.checkpoint_load)) # torch.load(path, map_location='cuda:0')
        print("Training from ", TRAIN_EPOCH_START, ", loading parameters from ./save/", args.checkpoint_load)
    else:
        TRAIN_EPOCH_START = 0

    model.train()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda() # gpu 0/1/2
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    
    loss_list = list()
    train_accuracy_list = list()
    val_accuracy_list = list()
    for epoch in range(TRAIN_EPOCH_START, EPOCH_TRAIN): # args.epochs
        sampler.set_epoch(epoch) # 每次epoch打乱顺序
        total_loss = 0 # 单个GUP上每个epoch的总loss
        total_num = 0 # 单个GUP上每个epoch的总num
        # total_right = 0 # 每个epoch在训练集上的总right个数
        train_acc = [] # 单个GUP上的数据在训练集上的accuracy list
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

        if((epoch+1)%20 == 0): # 每10个epoch输出/记录一次loss和accuracy
            model.eval()
            val_acc = []
            with torch.no_grad():
                for i, (image, label) in enumerate(val_dataloader):
                    val_outputs = model(image.cuda(non_blocking=True))
                    val_acc.append(accuracy_score(label.numpy(), torch.max(val_outputs, 1)[1].cpu().numpy()))
                print("Epoch: ", epoch+1, "; Loss: ", total_loss/total_num, "; Train Acuracy: ", np.mean(np.array(train_acc)), "Val Accuracy:", np.mean(np.array(val_acc)))
                loss_list.append(total_loss/total_num) # 每个sample上的loss
                train_accuracy_list.append(np.mean(np.array(train_acc)))
                val_accuracy_list.append(np.mean(np.array(val_acc)))
            model.train()

        if((epoch+1)%500 == 0):  # 每500个epoch存一次参数
            torch.save(model.state_dict(), "./save/"+args.model+"_"+str(epoch+1)+".pth")

    if args.checkpoint_save:
        torch.save(model.state_dict(), "./save/"+args.checkpoint_save)
    else:
        print("Automatically save the checkpoint into './save/MODEL_EPOCH_TRAIN.pth'")
        torch.save(model.state_dict(), "./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth")
    np.save("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_loss.npy", torch.tensor(loss_list, device = 'cpu'))
    np.save("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_train_accuracy.npy", torch.tensor(train_accuracy_list, device = 'cpu'))
    np.save("./save/"+args.model+"_"+str(EPOCH_TRAIN)+"_val_accuracy.npy", torch.tensor(val_accuracy_list, device = 'cpu'))
    plot(args, "loss")
    plot(args, "train_accuracy")
    plot(args, "val_accuracy")
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
        model.load_state_dict(torch.load("./save/"+args.checkpoint_load)) # torch.load(path, map_location='cuda:0')
    else:
        if os.path.isfile("./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth"):
            print("Automatically load the checkpoint from './save/MODEL_EPOCH_TRAIN.pth'")
            model.load_state_dict(torch.load("./save/"+args.model+"_"+str(EPOCH_TRAIN)+".pth")) # torch.load(path, map_location='cuda:0')
        else:
            print("Without using parameters.") # If trained, the program will automatically using training parameters.
        
    model.eval()
    # params = list(model.named_parameters())
    # print(params.__len__())
    # print(params[0])
    criterion = nn.CrossEntropyLoss().cuda() # gpu 0/1/2

    total_test_loss = 0
    total_accuracy = 0
    y_label = []
    y_pred = []
    y_person_id = []

    with torch.no_grad():
        # for i, (image1, image2, label, person_id) in enumerate(dataloader):
        for image, label in tqdm(dataloader):
            images = image.cuda(non_blocking=True)
            labels = label.cuda(non_blocking=True) 
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1)==labels).sum()
            total_accuracy = total_accuracy + accuracy
            y_label = y_label + label.tolist()
            y_pred = y_pred + outputs.argmax(1).tolist()

            output = outputs.cpu().numpy()
            target = label.cpu().numpy()
            # confusion.update(output_1[0], target_1[0])
    
            score_list.extend(output)
            label_list.extend(target) 

    nagetive_y_label = []
    nagetive_y_pred = []
    for i in range(len(y_label)):
        if y_label[i] != y_pred[i]:
            # print(y_label[i],y_pred[i])
            nagetive_y_label.append(y_label[i])
            nagetive_y_pred.append(y_pred[i])
    
    if score_array.size == 1:
        score_array = np.array(score_list)
    else:
        score_array = np.concatenate((score_array, np.array(score_list)))
    # label to onehot
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], NUM_CLASSES)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    if label_onehot_array.size == 1:
        label_onehot_array = np.array(label_onehot)
    else:
        label_onehot_array = np.concatenate((label_onehot_array, np.array(label_onehot)))
    print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
    print("label_onehot_array:", label_onehot_array.shape)  # torch.Size([batchsize, classnum]) onehot

    return nagetive_y_label, nagetive_y_pred, total_accuracy, y_label, y_pred


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
    if args.model == 'swin_OCT' or args.model == 'res_OCT':
        train_dataset = torchvision.datasets.ImageFolder('/home/.../data_OCT/'
                            ,transform=transforms.Compose([
                                                            transforms.Resize((224,224)),
                                                            transforms.ToTensor()
                                                        ]))
        test_dataset = torchvision.datasets.ImageFolder('/home/.../data_OCT/'
                            ,transform=transforms.Compose([
                                                            transforms.Resize((224,224)),
                                                            transforms.ToTensor()
                                                        ]))
        print(train_dataset.__len__())
        print(test_dataset.__len__())
    elif args.model == 'swin_fundus' or args.model == 'res_fundus':
        train_dataset = torchvision.datasets.ImageFolder('/home/.../data_fundus/'
                            ,transform=transforms.Compose([
                                                            transforms.Resize((224,224)),
                                                            transforms.ToTensor()
                                                        ]))
        test_dataset = torchvision.datasets.ImageFolder('/home/.../data_fundus/'
                            ,transform=transforms.Compose([
                                                            transforms.Resize((224,224)),
                                                            transforms.ToTensor()
                                                        ]))
        print(train_dataset.__len__())
        print(test_dataset.__len__())
    # train_dataset = ConcatDataset([train_dataset, test_dataset])

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset) # if is_distributed else None
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        test_sampler = DistributedSampler(test_dataset) # if is_distributed else None
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, sampler=test_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)


    # loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler)
    # for epoch in range(start_epoch, n_epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)
    #         # 在分布式模式下，需要在每个 epoch 开始时调用 set_epoch() 方法，然后再创建 DataLoader 迭代器，
    #         # 以使 shuffle 操作能够在多个 epoch 中normal工作。 否则，dataloader迭代器产生的数据将始终使用相同的顺序。
    #         train(train_dataloader)

    if args.model == 'swin_OCT':
        model = MyModel_swin_OCT(NUM_CLASSES)
    elif args.model == 'swin_fundus':
        model = MyModel_swin_fundus(NUM_CLASSES)
    elif args.model == 'res_OCT':
        model = MyModel_res_OCT(NUM_CLASSES)
    elif args.model == 'res_fundus':
        model = MyModel_res_fundus(NUM_CLASSES)
    else: 
        print("Model does not exit.")
        exit
    model = model.to(device)
    # if torch.cuda.device_count()>1: # Wrap the model
    model = nn.parallel.DistributedDataParallel(model) # gpu 0,1,2

    ############################################################
    if args.eval == 'False': # train
        train(args, model, train_sampler, train_dataloader, test_dataloader)

    ###########################################################
    y_label_list = []
    y_pred_list = []
    nagetive_tulpe_list = []
    accuracy = 0
    for epoch in range(EPOCH_TEST):
        print('Eval: ', epoch+1)
        test_sampler.set_epoch(epoch) # 每次epoch打乱顺序
        nagetive_y_label, nagetive_y_pred, total_accuracy, y_label, y_pred\
              = eval(args, model, test_dataloader)
        #########################
        dist.barrier()  # synchronizes all processes
        dist.reduce(total_accuracy, op=torch.distributed.ReduceOp.SUM, dst=0)    
        nagetive_y_label_all_gpu = nagetive_y_pred_all_gpu = y_label_all_gpu = y_pred_all_gpu \
            = [None for _ in range(torch.cuda.device_count())] # 申明
        dist.all_gather_object(nagetive_y_label_all_gpu, nagetive_y_label)
        nagetive_y_label_all_gpu = [i for ii in nagetive_y_label_all_gpu for i in ii] # 降维
        dist.all_gather_object(nagetive_y_pred_all_gpu,  nagetive_y_pred)
        nagetive_y_pred_all_gpu = [i for ii in  nagetive_y_pred_all_gpu for i in ii] # 降维
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

        y_label_list = y_label_list + y_label_all_gpu
        y_pred_list = y_pred_list + y_pred_all_gpu
        accuracy = accuracy + total_accuracy
    nagetive = Counter(nagetive_tulpe_list)
    d_nagetive = sorted(nagetive.items(), key=lambda x: x[1], reverse=True)

    plt.cla()
    confusion_mat = confusion_matrix(y_label_list, y_pred_list)
    # classes = ["DR", "AMD", "HM", "RVO", "GLC", "normal"]
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat) # 傻逼玩意儿版本不对，自定义标签值能生成图片但会报错
    disp.plot(
        include_values=True,
        cmap="viridis",
        ax=None,
        xticks_rotation="horizontal",
        values_format="d"     
    )
    plt.savefig("./save/CM_"+args.model+"_"+str(EPOCH_TRAIN)+".png")

print(classification_report(y_label_list, y_pred_list, digits=4))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(label_onehot_array[:, i], score_array[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(label_onehot_array.ravel(), score_array.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
lw = 2
for i, color in zip(range(NUM_CLASSES), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.4f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.savefig("./ROC.png")

print("micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]))
print("macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]))