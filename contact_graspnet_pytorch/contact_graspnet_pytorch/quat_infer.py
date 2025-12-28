from genericpath import exists
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # To get pyrender to work headless

# Import pointnet library
CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Pointnet_Pointnet2_pytorch'))

import config_utils
from acronym_dataloader import AcryonymDataset
from contact_graspnet_pytorch.contact_graspnet import ContactGraspnet, ContactGraspnetLoss
from contact_graspnet_pytorch import utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO 
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# ckpt_dir = "checkpoints/contact_graspnet"
# ckpt_dir = "checkpoints/test_training"
# ckpt_dir = "checkpoints/test_4_point_training"
ckpt_dir = "checkpoints/gmm-no-sigmoid"
data_path = "/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/contact_graspnet/acronym"
batch_size = 6
global_config = config_utils.load_config(ckpt_dir, batch_size=batch_size, data_path=data_path, save=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### set up the test dataset
num_workers = 0  # Increase after debug
global_config['DATA']['num_test_scenes'] = 100
test_dataset = AcryonymDataset(global_config, train=False, device=device, use_saved_renders=True, eval_with_fixed_cam=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers)

grasp_estimator = GraspEstimator(global_config)

## TODO: change this to be computing the loss for the 4 points
loss_fn = ContactGraspnetLoss(global_config, device).to(device)

### load the pretrained model
model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
load_dict = checkpoint_io.load('model_best.pt')
# load_dict = checkpoint_io.load('model.pt')
# load_dict = checkpoint_io.load('model_30000.pt')
grasp_network = grasp_estimator.model

grasp_network.eval()
with torch.no_grad():
    loss_log = []
    topk_4_points_loss = []
    for val_it, data in enumerate(tqdm(test_dataloader)):
        # print("Validation iteration: ", val_it)
        
        utils.send_dict_to_device(data, device)
        # Target contains input and target values
        pc_cam = data['pc_cam']
        # import pdb; pdb.set_trace()

        pred = grasp_network(pc_cam)
        ### TODO: change the loss here
        loss, loss_info = loss_fn(pred, data, compute_topk_4_points_loss=True)
        # loss_log.append(loss.item())
        # loss_log.append(loss_info['adds_loss'].item())
        loss_log.append(loss_info['four_point_loss'].item())
        topk_4_points_loss.append(loss_info['topk_4_point_loss'].item())
        
    val_loss = np.mean(loss_log)

print(f"{ckpt_dir}: Validation loss: {val_loss:.4f} topk 4 point loss: {np.mean(topk_4_points_loss):.4f} ")    


