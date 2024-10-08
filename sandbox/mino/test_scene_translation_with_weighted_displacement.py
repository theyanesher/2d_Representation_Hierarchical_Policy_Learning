"""
Tests the high-level policy with weighted displacement for scene translation invariance. 
"""
import random
import torch 

from sandbox.mino.test_scene_translation import load_train_dataset, get_dataloader, \
                                        set_random_seed, translate_batch
from diffusion_policy_3d.common.pytorch_util import dict_apply
from test_PointNet2.model import PointNet2

def load_high_level_weighted_displacement_policy():
    load_model_path = "/home/mino/Software/RoboGen-sim2real/test_PointNet2/exps/model_36.pth"
    pointnet2_model = PointNet2(num_classes=13).to('cuda')
    pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    return pointnet2_model

def run_high_level_policy_inference(policy, batch):
    pointcloud = batch['point_cloud'][:, -1, :, :]
    gripper_pcd = batch['gripper_pcd'][:, -1, :]
    inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    outputs = policy(inputs_)
    weights = outputs[:, :, -1] # B, N
    outputs = outputs[:, :, :-1] # B, N, 12
    B, N, _ = outputs.shape
    outputs = outputs.view(B, N, 4, 3)
    outputs = outputs + inputs.unsqueeze(2)
    weights = torch.nn.functional.softmax(weights, dim=1)
    outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
    outputs = outputs.sum(dim=1)
    outputs = outputs.unsqueeze(1)
    return outputs


def test_translation_invariance_with_weighted_displacement(policy, dataloader):
    print("Calculating difference between outputs")
    norms = []
    device = torch.device(0)
    for i, batch in enumerate(dataloader):
        if i == 20:
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))['obs']
        
        seed = random.randint(0, 100)
        # set_random_seed(seed, using_cuda=True)
        model_output = run_high_level_policy_inference(policy, batch)

        translated_batch = batch
        translation_vector = torch.ones(3).to('cuda:0')
        translated_batch = translate_batch(batch, translation_vector)
        set_random_seed(seed, using_cuda=True)
        translated_model_output = run_high_level_policy_inference(policy, translated_batch)
        diff = torch.linalg.norm(model_output - translated_model_output, axis=-1)
        # breakpoint()
        norms.append(diff)
        del model_output, translated_model_output
    average_diff = torch.mean(torch.stack(norms))
    print(f"diff: {average_diff}")
    return average_diff

def main():

    highlevel_policy = load_high_level_weighted_displacement_policy()
    train_dataset = load_train_dataset()
    train_dataloader = get_dataloader(train_dataset, batch_size=2)

    test_translation_invariance_with_weighted_displacement(highlevel_policy, train_dataloader)
    
if __name__ == '__main__':
    main()