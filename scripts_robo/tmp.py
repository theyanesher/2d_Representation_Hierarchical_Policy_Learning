import os


list_of_new_objs = [
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-41003:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45001:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45235:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45238:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45244:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45249:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0705-obj-45523:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0705-obj-46014:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0705-obj-46166:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0705-obj-46653:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0705-obj-47711:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0705-obj-48263:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45007:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45087:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45159:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45166:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45189:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45247:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45261:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45267:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-45354:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45413:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45420:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45594:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45670:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45916:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-45950:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-46092:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-46134:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-46230:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-46277:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-46334:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-46443:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-46466:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-46480:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-46641:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-47088:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-47185:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-47254:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-47419:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-47613:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0725-obj-47742:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-48018:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-48023:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-48051:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-48271:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-48491:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-48519:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-48740:/jet/projects/cis240052p/ywang59/dp3_demo/"
"/ocean/projects/cis240052p/ywang59/RoboGen_sim2real/data/dp3_demo_pickle/0730-obj-49140:/jet/projects/cis240052p/ywang59/dp3_demo/"
"0822-obj-1003",
"0822-obj-1014",
"0822-obj-1014",
"0822-obj-1065",
"0822-obj-1079",
"0822-obj-1086",
"0822-obj-1094",
"0822-obj-1121",
"0822-obj-1166",
"0822-obj-1170",
"0822-obj-1204",
"0822-obj-1204",
"0822-obj-1225",
"0822-obj-1248",
"0822-obj-1253",
"0822-obj-1253",
"0822-obj-1254",
"0822-obj-1255",
"0822-obj-1255",
"0822-obj-1255",
"0822-obj-1256",
"0822-obj-1256",
"0822-obj-1256",
"0822-obj-1257",
"0822-obj-1258",
"0822-obj-1258",
"0822-obj-1259",
"0822-obj-1259",
"0822-obj-1259",
"0822-obj-1259",
"0822-obj-1260",
"0822-obj-1260",
"0822-obj-1261",
"0822-obj-1261",
"0822-obj-7119",
"0822-obj-7167",
"0822-obj-7187",
"0822-obj-7220",
"0822-obj-7263",
"0822-obj-7290",
]

new_idx_start = 197
for new_obj in list_of_new_objs:
    print("save_data_name_{}={}".format(new_idx_start, new_obj))
    new_idx_start += 1
    


