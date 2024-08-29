import os
# os.system("export PYTHONPATH=${PWD}:PYTHONPATH")

import time
import datetime
import json
from chester.run_exp import VariantGenerator

seuss_user = 'yufeiw2'
seuss_project_folder = 'RoboGen_sim2real'

def vv_to_params_seuss(vv):
    params = "{} {} {} {}".format(
        vv['exp_folder'], vv['save_data_name'], vv['demo_name'], vv['folder_name']
    )
        
    print("running params: ", params)
    return params

def run_task(vv):
    print("vv: ", vv)
        
    params = vv_to_params_seuss(vv)
    
    log_file = '/data/yufeiw2/RoboGen_sim2real/data/local/gen_data.log'
    exp_name = vv['exp_name']
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')

    
    log_dir = os.path.join("/data/{}/{}/".format(seuss_user, seuss_project_folder), "data/local", vv['exp_folder'])
    out_log = os.path.join(log_dir, 'stdout.log')
    err_out = os.path.join(log_dir, 'stdout.err')

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=4, sort_keys=True)

    command = "sbatch -o {} -e {} -J {} scripts/sbatch_gen_data.sh {}".format(
        out_log, err_out, vv['exp_folder'],  params)
    # command = "sbatch scripts/sbatch_gen_data.sh {}".format(
        # params)
    print(command)
    os.system(command)
    time.sleep(5)

def get_save_data_name(exp_folder):
    return "0730-obj-" + str(exp_folder.split("_")[-1])

# generate all parameter combinations you want to test
vg = VariantGenerator()
vg.add("exp_name", ["test-gen-data"])

vg.add("exp_folder", [
    # "open_the_door_45413", 
    # "open_the_door_45420", 
    # "open_the_door_45427", 
    # "open_the_door_45443", 
    # "open_the_door_45504", 
    # "open_the_door_45594", 
    # "open_the_door_45620", 
    # "open_the_door_45623", 
    # "open_the_door_45633", 
    # "open_the_door_45636", 
    # "open_the_door_45667", 
    
    
    
    # "open_the_door_45767", 
    # "open_the_door_45841", 
    # "open_the_door_45916", 
    # "open_the_door_45922", 
    # "open_the_door_45936", 
    # "open_the_door_45937", 
    # "open_the_door_45950", 
    # "open_the_door_45984", 
    # "open_the_door_46092", 
    
    
    # "open_the_door_45670",
    # "open_the_door_46107",
    # "open_the_door_46109",
    # "open_the_door_46130",
    # "open_the_door_46134",
    # "open_the_door_46197",
    # "open_the_door_46334",
    # "open_the_door_46401",
    # "open_the_door_46443",
    # "open_the_door_46456",
    
    ## batch 3
    # "open_the_door_47178",
    # "open_the_door_47180",
    # "open_the_door_47182",
    # "open_the_door_47187",
    # "open_the_door_47227",
    # "open_the_door_47238",
    # "open_the_door_47254",
    # "open_the_door_47466",
    # "open_the_door_47565",
    # "open_the_door_47577",
    # "open_the_door_47648",
    
    ## batch 4
    # "open_the_door_47742",
    # "open_the_door_47747",
    # "open_the_door_47808",
    # "open_the_door_47817",
    # "open_the_door_47954",
    # "open_the_door_47963",
    # "open_the_door_47976",
    # "open_the_door_48010",
    # "open_the_door_48013",
    # "open_the_door_48036",
    # "open_the_door_48258",
    # "open_the_door_48379",
    # "open_the_door_48381",
    # "open_the_door_48797",
    # "open_the_door_48855",
    # "open_the_door_48859",
    # "open_the_door_49042",
    # "open_the_door_49182",
    # "open_the_door_49188",
    
    # batch 5
    # "open_the_door_48258",
    # "open_the_door_48379",
    # "open_the_door_46130",
    # "open_the_door_45620",
    # "open_the_door_47178",
    
    # "open_the_door_45937", 
    # "open_the_door_46334" ,
    # "open_the_door_46443" ,
    # "open_the_door_47254" ,
    # "open_the_door_49042" ,
    
    
    ### 0730 all rest objects
    # "open_the_door_35059",
    # "open_the_door_41004",
    # "open_the_door_41083",
    # "open_the_door_41529",
    # "open_the_door_44781",
    # "open_the_door_44826",
    # "open_the_door_44853",
    # "open_the_door_45007",
    # "open_the_door_45087",
    # "open_the_door_45091",
    # "open_the_door_45092",
    # "open_the_door_45130",
    # "open_the_door_45134",
    # "open_the_door_45135",
    # "open_the_door_45146",
    # "open_the_door_45159",
    # "open_the_door_45164",
    
    ### 0730 all rest objects batch 2
#     "open_the_door_45166", 
#  "open_the_door_45168", 
#  "open_the_door_45173", 
#  "open_the_door_45177", 
#  "open_the_door_45189", 
#  "open_the_door_45212", 
#  "open_the_door_45213", 
#  "open_the_door_45247", 
#  "open_the_door_45261", 
#  "open_the_door_45267", 
#  "open_the_door_45354", 
#  "open_the_door_45372", 
#  "open_the_door_45374", 
#  "open_the_door_45385", 
#  "open_the_door_45387", 
#  "open_the_door_45403", 
#  "open_the_door_45415", 
#  "open_the_door_45419", 
#  "open_the_door_45423", 
#  "open_the_door_45503", 
#  "open_the_door_45505", 
#  "open_the_door_45524", 
#  "open_the_door_45573", 
#  "open_the_door_45575", 
#  "open_the_door_45606", 
#  "open_the_door_45612", 
#  "open_the_door_45621", 
#  "open_the_door_45622", 
#  "open_the_door_45632", 
#  "open_the_door_45638", 
#  "open_the_door_45642", 
#  "open_the_door_45645", 
#  "open_the_door_45662", 
#  "open_the_door_45671", 
#  "open_the_door_45676", 
#  "open_the_door_45677", 
#  "open_the_door_45687", 
#  "open_the_door_45699", 
#  "open_the_door_45710", 
#  "open_the_door_45746", 
#  "open_the_door_45756", 
#  "open_the_door_45776", 
#  "open_the_door_45779", 
#  "open_the_door_45783", 
#  "open_the_door_45784", 
#  "open_the_door_45790", 
#  "open_the_door_45801", 
#  "open_the_door_45822", 
#  "open_the_door_45853", 
#  "open_the_door_45855", 
#  "open_the_door_45908", 
#  "open_the_door_45915", 
#  "open_the_door_45940", 
#  "open_the_door_45948", 
#  "open_the_door_45949", 
#  "open_the_door_45963", 
#  "open_the_door_45964", 
#  "open_the_door_46002", 
#  "open_the_door_46019", 
#  "open_the_door_46029", 
#  "open_the_door_46033", 
#  "open_the_door_46037", 
#  "open_the_door_46044", 
#  "open_the_door_46045", 
#  "open_the_door_46060", 
#  "open_the_door_46084", 
#  "open_the_door_46108", 
#  "open_the_door_46117", 
#  "open_the_door_46120", 
#  "open_the_door_46123", 
#  "open_the_door_46132", 
#  "open_the_door_46145", 
#  "open_the_door_46179", 
#  "open_the_door_46180", 
#  "open_the_door_46199", 
#  "open_the_door_46230", 
#  "open_the_door_46277", 
#  "open_the_door_46380", 
#  "open_the_door_46427", 
#  "open_the_door_46430", 
#  "open_the_door_46439", 
#  "open_the_door_46452", 
#  "open_the_door_46466", 

### 0730 all rest objects batch 3
 "open_the_door_46537",
"open_the_door_46549",
"open_the_door_46556",
"open_the_door_46598",
"open_the_door_46616",
"open_the_door_46699",
"open_the_door_46700",
"open_the_door_46741",
"open_the_door_46744",
"open_the_door_46847",
"open_the_door_46856",
"open_the_door_46859",
"open_the_door_46889",
"open_the_door_46906",
"open_the_door_46944",
"open_the_door_46955",
"open_the_door_46981",
"open_the_door_47021",
"open_the_door_47024",
"open_the_door_47088",
"open_the_door_47089",
"open_the_door_47183",
"open_the_door_47185",
"open_the_door_47207",
"open_the_door_47233",
"open_the_door_47252",
"open_the_door_47278",
"open_the_door_47290",
"open_the_door_47296",
"open_the_door_47388",
"open_the_door_47391",
"open_the_door_47419",
"open_the_door_47438",
"open_the_door_47514",
"open_the_door_47585",
"open_the_door_47595",
"open_the_door_47601",
"open_the_door_47613",
"open_the_door_47632",
"open_the_door_47701",
"open_the_door_47729",
"open_the_door_47853",
"open_the_door_47926",
"open_the_door_48018",
"open_the_door_48023",
"open_the_door_48051",
"open_the_door_48271",
"open_the_door_48413",
"open_the_door_48452",
"open_the_door_48467",
"open_the_door_48490",
"open_the_door_48491",
"open_the_door_48513",
"open_the_door_48517",
"open_the_door_48519",
"open_the_door_48686",
"open_the_door_48721",
"open_the_door_48740",
"open_the_door_48746",
"open_the_door_48878",
"open_the_door_49140",
 
])
vg.add("save_data_name", lambda exp_folder: [get_save_data_name(exp_folder)])
vg.add("demo_name", ["0730-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"])
vg.add("folder_name", ["data/diverse_objects_rest/"])

# for each parameter combination, run the epxeriment
all_vvs = vg.variants()
for vv in all_vvs:
    run_task(vv)