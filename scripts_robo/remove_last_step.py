import os

path = "/jet/projects/cis240052p/ywang59/dp3_demo/"
all_folder = os.listdir(path)
for folder in all_folder:
    all_traj = os.listdir(os.path.join(path, folder))
    for traj in all_traj:
        traj_path = os.path.join(path, folder, traj)
        all_steps = os.listdir(traj_path)
        last_step = sorted(all_steps, key=lambda x: int(x.split(".")[0]))[-1]
        last_step_path = os.path.join(traj_path, last_step)
        print("Removing", last_step_path)
        # import pdb; pdb.set_trace()
        os.system("rm -r {}".format(last_step_path))