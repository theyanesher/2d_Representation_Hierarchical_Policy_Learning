import sys
import os
import argparse

sys.path.append('.')
from chester import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('host', type=str)
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    parser.add_argument('--img', action='store_true', default=False)
    parser.add_argument('--pkl', action='store_true', default=False)
    parser.add_argument('--pth', action='store_true', default=False)
    parser.add_argument('--gif', action='store_true', default=False)
    parser.add_argument('--best', action='store_true', default=False)
    parser.add_argument('--reward', action='store_true', default=False)
    parser.add_argument('--policy', action='store_true', default=False)
    args = parser.parse_args()

    local_dir = os.path.join('./exp', args.host, args.exp_name)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    print("pulling from {} {}".format(args.host, args.exp_name))

    if args.host == 'seuss':
        dir_path = '/data/yufeiw2/RoboGen_sim2real/'
    elif args.host == 'autobot':
        dir_path = '/project_data/held/yufeiw2/RoboGen_sim2real/'

    remote_data_dir = os.path.join(dir_path, 'exp', args.exp_name)
    command = """rsync -avzh --progress {host}:{remote_data_dir} {local_dir} --include '*best_model.pth'  """.format(host=args.host,
                                                                                                remote_data_dir=remote_data_dir,
                                                                                                local_dir=local_dir)
    print(command)
                                                                                            
    if not args.img:
        command += """ --exclude '*.png' """
    if not args.gif:
        command += """ --exclude '*.gif' """
    # command += """  --exclude '*.pth' """

    if args.best:
        command += """ --include '*best*.pt'  """
        command += """ --include '*best*.pkl'  """
        command += """ --include '*best*.pth'  """

    if args.reward:
        command += """ --include '*reward*1000000*.pt'  """
    if args.policy:
        command += """ --include '*1000000*.pt'  """

    if not args.pkl:
        command += """ --exclude '*.pkl'  """
    

    if args.bare:
        command += """  --exclude '*checkpoint*' --exclude '*ckpt*' --exclude '*wandb*' --exclude '*.pth' --exclude '*.mp4'  --exclude '*tfevents*' --exclude '*.pt' --include '*.csv' --include '*.json' --delete"""



    # if args.dry:
    print(command)
    # else:
    os.system(command)
