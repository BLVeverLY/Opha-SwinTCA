import os
import argparse

############ Set it correctly for distributed training across nodes
NNODES = 1  # e.g. 1/2/3/4
NPROC_PER_NODE = 3  # e.g. 3 gpus

MASTER_ADDR = 'SET_IT' # master_addr 和 master_port 是 master 监听的地址和端口
MASTER_PORT = 12345
NODE_RANK = 0  # e.g. 0/1/2
############

print("NNODES, ", NNODES)
print("NPROC_PER_NODE, ", NPROC_PER_NODE)
print("MASTER_ADDR, ", MASTER_ADDR)
print("MASTER_PORT, ", MASTER_PORT)
print("NODE_RANK, ", NODE_RANK)


def get_dist_launch(args):  # some examples
    if args.dist == 'all':  # use all nodes all gpus, NODE_RANK = e.g. 0/1/2/...
        # return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
        #        "--nnodes={:} --node_rank={:} --master_addr={:} --master_port={:} --use_env" \
        #         .format(NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)
        return "torchrun --nproc_per_node={:} --nnodes={:} --node_rank={:} --master_addr={:} " \
               "--master_port={:}".format(NPROC_PER_NODE, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT)

    elif args.dist == '1': # single-node multi-worker
        # return "python3 -m torch.distributed.launch --nproc_per_node={:} " \
        #        "--nnodes=1 --use_env ".format(NPROC_PER_NODE)
        return "torchrun --standalone --nproc_per_node={:} --nnodes=1".format(NPROC_PER_NODE)
        # --standalone 啥意思实在不知道，盲猜是不需要手动设置进程端口

    elif args.dist == 'f3': # 1 machine, first 3 gpus
        # return "CUDA_VISIBLE_DEVICES=0,1,2 WORLD_SIZE=3 python3 -m torch.distributed.launch --nproc_per_node=4 " \
        #        "--nnodes=1 --use_env "
        return "CUDA_VISIBLE_DEVICES=0,1,2 WORLD_SIZE=3 torchrun --nproc_per_node=3 --nnodes=1"
        # WORLD_SIZE: 总的进程数量，此处设置为一个process占用一个GPU (== NNODES); world_size = gpus * nodes

    elif args.dist == 'l4': # 1 machine, last 4 gpus
        # return "CUDA_VISIBLE_DEVICES=4,5,6,7 WORLD_SIZE=4 python3 -m torch.distributed.launch --master_port=12345 " \
        #        "--nproc_per_node=4 --nnodes=1 --use_env "
        return "CUDA_VISIBLE_DEVICES=4,5,6,7 WORLD_SIZE=4 torchrun --nproc_per_node=4 --nnodes=1"

    elif args.dist.startswith('gpu'):  # 1 machine, 1 gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= NPROC_PER_NODE
        # return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.launch --nproc_per_node=1 " \
        #        "--nnodes=1 --use_env ".format(num)
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 torchrun --nproc_per_node=1 --nnodes=1 ".format(num)

    else:
        raise ValueError


if __name__ == '__main__':
    parser = argparse.ArgumentParser('My project', add_help=False)
    parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")
    parser.add_argument('--model', type=str, required=True, help="swin or resnet, cat or trans")
    # parser.add_argument('--config', default='./configs/train.yaml', type=str, help="if not given, use default")
    # parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', type=int, default=80,help="batch size for each gpu during training or evaluating;"
                                                        "the batch size in total equals batch_size*NPROC_PER_NODE*NNODES")
    parser.add_argument('--checkpoint_save', default='', type=str, help="local path for parameters.") # after train
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--checkpoint_load', default='', type=str, help="local path for parameters") # before train or eval
    args, unparsed = parser.parse_known_args() # Redundant and useless commands are allowed
    # args = parser.parse_args()

    if MASTER_ADDR == 'SET_IT':
        print("Warning: the settings for distributed training is not filled (ignore this if you only use one node)")

    dist_launch = get_dist_launch(args)
    cmd = "{dl} run.py --model {m} --batch_size {bs} {ckps} {e} {ckpl}" \
        .format(dl=dist_launch, m=args.model, bs=args.batch_size, \
                ckps=" --checkpoint_save "+args.checkpoint_save if args.checkpoint_save!="" else "", \
                e=" --eval True " if args.eval!=False else "--eval False ", \
                ckpl=" --checkpoint_load "+args.checkpoint_load if args.checkpoint_load!="" else "")
                # --config {args.config}
    print(cmd)
    os.system(cmd)
             
