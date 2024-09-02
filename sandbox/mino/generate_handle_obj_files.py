import functools
import json 
import multiprocessing as mp
from pathlib import Path

from manipulation.scripts.extract_handle_mesh import main

REPO_ROOT = Path('/home/mino/Software/RoboGen-sim2real')
MOBILITY_DICT = Path('data/partnet_mobility_dict.json')
DATASET = Path('data/dataset')

def load_partnet_mobility_dict(): 
    mobility_dict_path = REPO_ROOT / MOBILITY_DICT
    with open(mobility_dict_path, 'r') as json_data:
        return json.load(json_data)

def process_category_id(category, id):
    if (REPO_ROOT / DATASET / id / "parts_render").exists():
        main(category, id)

def parse_all_objects(): 
    mobility_dict = load_partnet_mobility_dict()
    for category, ids in mobility_dict.items():
        print(f"processing object {category}")
        pool = mp.Pool(processes = 10)
        category_fn = functools.partial(process_category_id, category)
        pool.map(category_fn, ids)

parse_all_objects()