import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import zarr

class PointNetDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, gripper_points=None):
        self.data = data
        self.labels = labels
        self.gripper_points = gripper_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, gripper_pcd=False):
        return self.data[idx], self.labels[idx], self.gripper_points[idx]

def get_dataloader(zarr_path=None, batch_size=32, beg_ratio=0, end_ratio=0.9, shuffle=True):
    point_cloud, gripper_pcd, binary_mask = load_from_zarr(zarr_path)
    length = len(point_cloud)
    beg = int(beg_ratio * length)
    end = int(end_ratio * length)
    data = point_cloud[beg:end]
    labels = binary_mask[beg:end]
    gripper_points = gripper_pcd[beg:end]
    dataset = PointNetDataset(data, labels, gripper_points)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def visualize_pointcloud(pointcloud, gripper_pcd=None, binary_mask=None):
    true_points = pointcloud[binary_mask == 1]
    false_points = pointcloud[binary_mask == 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], c='r', marker='o',s=10)
    ax.scatter(false_points[:, 0], false_points[:, 1], false_points[:, 2], c='b', marker='o',s=1)
    if gripper_pcd is not None:    
        ax.scatter(gripper_pcd[:,0], gripper_pcd[:,1], gripper_pcd[:,2], c='g', marker='o',s=3)
    plt.show()


def load_from_zarr(zarr_path=None):
    if zarr_path is None:
        zarr_path = "/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/data"
    group = zarr.open(zarr_path, 'r')
    src_store = group.store
    src_root = zarr.group(src_store)
    point_cloud = src_root['data']['point_cloud'][:]
    gripper_pcd = src_root['data']['gripper_pcd'][:]
    binary_mask = src_root['data']['binary_mask'][:]

    return point_cloud, gripper_pcd, binary_mask



if __name__ == "__main__":
    point_cloud, gripper_pcd, binary_mask = load_from_zarr()
    visualize_pointcloud(point_cloud[0], gripper_pcd[0], binary_mask[0])