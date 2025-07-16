import os
# os.system("export PYTHONPATH=${PWD}:PYTHONPATH")

import time
import json
from chester.run_exp import VariantGenerator

def vv_to_params_seuss(vv):
    params = "{} {}".format(
        vv['folder_name'], vv['exp_folder'],
    )
    print("running params: ", params)
    return params

def run_task(vv):
    print("vv: ", vv)
        
    params = vv_to_params_seuss(vv)

    log_dir = os.path.join("/data/chenyuah/RoboGen-sim2real/sbatch_logs", vv['exp_folder'])
    out_log = os.path.join(log_dir, 'stdout.log')
    err_out = os.path.join(log_dir, 'stdout.err')

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=4, sort_keys=True)

    command = "sbatch -o {} -e {} -J {} scripts/sbatch_dagger.sh {}".format(
        out_log, err_out, vv['exp_folder'],  params)

    print(command)
    os.system(command)
    time.sleep(5)

bucket_tasks = [100443, 100444, 100452, 100454, 100460, 100461, 100462, 100469, 100472, 102352, 102358, 102365]
faucet_tasks = [149, 152, 153, 154, 168, 811, 822, 857, 908, 929, 1028, 1052, 1053, 1288, 1343, 1370, 1466, 1492, 1528, 1626, 1633, 1646, 1668, 1741, 1794, 1795, 1802, 1885, 1901, 1903, 1925,  1961, 1986, 2054]
foldingchair_tasks = [100531, 100532, 100557, 100561, 100562, 100568, 100579, 100586, 100590, 100599, 100600, 100608, 100609, 100611, 100616, 102255, 102263, 102269, 102314]
laptop_tasks = [9968, 9992, 9996, 10040, 10098, 10101, 10238, 10243, 10248,10269, 10270, 10280, 10289, 10305, 10306, 10383, 10626, 10697, 10885, 10915, 11075, 11156, 11242, 11248, 11395, 11405, 11406, 11429, 11477, 11581, 11586, 11691, 11778, 11876, 11888, 11945, 12073]
stapler_tasks = [103099, 103100, 103104, 103111, 103113, 103271, 103275, 103276, 103280, 103292, 103293, 103297, 103299, 103301, 103303, 103305, 103789, 103792]
toilet_tasks = [102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654, 102658, 102663, 102666, 102667, 102668, 102669, 102670, 102675, 102676, 102677, 102687, 102689, 102692, 102694, 102697, 102699, 102701, 102703, 102707, 102708, 103234]

storagefurniture_tasks = [46874, 46922, 46966, 47570, 47578, 48700]

diverse_objects = ['41510', '45448', '46462', '46732', '46801', '46874', '46922', '46966', '47570', '47578', '48700', '45526', '45661', '45694', '45780', '45910', '45961', '46408', '46417', '46440', '46490', '46762', '46825', '46893', '47235', '47281', '47315', '47529', '47669', '47944', '48063', '48177', '48356', '48623', '48876', '49025', '49062', '49132', '49133', '40417', '41085', '41452', '45162', '45176', '45194', '45203', '45248', '45271', '45290', '45305', '45427', '45620', '45623', '45636', '45689', '45696', '45749', '45759', '45936', '45984', '46130', '46197', '46481', '46544', '47178', '47182', '47227', '47577', '47648', '47747', '47808', '47976', '48010', '48258', '48379', '48797', '48855', '48859', '49188', '35059', '41004', '41083', '44781', '44826', '44853', '45092', '45130', '45135', '45146', '45164', '45168', '45173', '45212', '45213', '45372', '45374', '45387', '45415', '45419', '45423', '45503', '45505', '45524', '45573', '45575', '45606', '45612', '45621', '45622', '45632', '45638', '45645', '45662', '45671', '45676', '45677', '45687', '45699', '45710', '45746', '45756', '45783', '45784', '45790', '45801', '45822', '45853', '45855', '45915', '45948', '45949', '45963', '45964', '46019', '46029', '46033', '46037', '46044', '46045', '46060', '46084', '46108', '46117', '46120', '46123', '46145', '46179', '46180', '46199', '46380', '46427', '46430', '46439', '46537', '46549', '46556', '46598', '46616', '46699', '46700', '46741', '46744', '46847', '46856', '46859', '46889', '46906', '46944', '46955', '46981', '47024', '47089', '47183', '47207', '47233', '47252', '47278', '47290', '47296', '47438', '47514', '47595', '47601', '47632', '47701', '47729', '47853', '47926', '48413', '48452', '48467', '48490', '48513', '48517', '48721', '48746', '48878', '41003', '45001', '45235', '45238', '45244', '45249', '45523', '46014', '46166', '46653', '47711', '48263', '45007', '45087', '45159', '45166', '45189', '45247', '45261', '45267', '45354', '45413', '45420', '45594', '45670', '45916', '45950', '46092', '46134', '46230', '46277', '46334', '46443', '46466', '46480', '46641', '47088', '47185', '47254', '47419', '47613', '47742', '48018', '48023', '48051', '48271', '48491', '48519', '48740', '10036', '10068', '10143', '10144', '10489', '10638', '10655', '10685', '10751', '10797', '10867', '10944', '11178', '11211', '11304', '11550', '11622', '11661', '11700', '11712', '11826', '12036', '12042', '12043', '12054', '12065', '12085', '12092', '12250', '12252', '12259', '12414', '12428', '12480', '12484', '12530', '12531', '12536', '12540', '12543', '12552', '12553', '12559', '12560', '12561', '12562', '12563', '12565', '12579', '12580', '12583', '12587', '12590', '12592', '12594', '12596', '12597', '12605', '12606', '12614', '12617', '7119', '7120', '7167', '7179', '7187', '7201', '7220', '7263', '7290', '7310', '7332']
bucket_tasks = [str(i) for i in bucket_tasks]
faucet_tasks = [str(i) for i in faucet_tasks]
foldingchair_tasks = [str(i) for i in foldingchair_tasks]
laptop_tasks = [str(i) for i in laptop_tasks]
stapler_tasks = [str(i) for i in stapler_tasks]
toilet_tasks = [str(i) for i in toilet_tasks]
storagefurniture_tasks = [str(i) for i in storagefurniture_tasks]
# diverse_objects = [str(i) for i in diverse_objects]
# diverse_objects_2 = [str(i) for i in diverse_objects_2]

# generate all parameter combinations you want to test
# vg = VariantGenerator()

# vg.add("folder_name", ["data/bucket/"])
# vg.add("exp_folder", bucket_tasks)

# # for each parameter combination, run the epxeriment
# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/faucet/"])
# vg.add("exp_folder", faucet_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/foldingchair/"])
# vg.add("exp_folder", foldingchair_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/laptop/"])
# vg.add("exp_folder", laptop_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/stapler/"])
# vg.add("exp_folder", stapler_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/toilet/"])
# vg.add("exp_folder", toilet_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
    # run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/storagefurniture/"])
# vg.add("exp_folder", storagefurniture_tasks)

# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)

vg = VariantGenerator()
vg.add("folder_name", ["data/diverse_objects_all/"])
vg.add("exp_folder", diverse_objects)

all_vvs = vg.variants()
for vv in all_vvs:
    run_task(vv)

# vg = VariantGenerator()
# vg.add("folder_name", ["data/diverse_objects_2/"])
# vg.add("exp_folder", diverse_objects_2)
# all_vvs = vg.variants()
# for vv in all_vvs:
#     run_task(vv)