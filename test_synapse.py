import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_synapse import Synapse_dataset
from utils.utils import test_single_volume

from REUNet.lib.networks import RMEUNet


parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='./data/synapse/train_npz', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
# network related parameters
parser.add_argument('--encoder', type=str,
                    default='resnet50', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--lgag_ks', type=int,
                    default=3, help='Kernel size in LGAG')
parser.add_argument('--activation_mscb', type=str,
                    default='relu6', help='activation used in MSCB: relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true',
                    default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true',
                    default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--no_pretrain', action='store_true',
                    default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')

parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=6, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--is_savenii', default=True,action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as nii!')
args = parser.parse_args()

if(args.num_classes == 14):
    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'portal vein and splenic vein', 'pancreas', 'right adrenal gland', 'left adrenal gland']
else:
    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class (%d) %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, classes[i-1], metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_jacard = np.mean(metric_list, axis=0)[2]
    mean_asd = np.mean(metric_list, axis=0)[3]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
    return "Testing Finished!"


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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'RMEUNet_' + dataset_name + str(args.img_size)

    net = RMEUNet(num_classes=args.num_classes, kernel_sizes=args.kernel_sizes, expansion_factor=args.expansion_factor,
             dw_parallel=not args.no_dw_parallel, add=not args.concatenation, lgag_ks=args.lgag_ks,
             activation=args.activation_mscb, encoder=args.encoder, pretrain=not args.no_pretrain).cuda()
    snapshot_path = ''

    snapshot = ''

    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(snapshot_path, "predictions")
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)