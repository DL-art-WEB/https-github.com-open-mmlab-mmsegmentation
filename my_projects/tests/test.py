import os
from mmengine import Config
from mmengine.analysis.complexity_analysis import (
    parameter_count_table, 
    parameter_count
)
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS

cfg_list = [
    "configs/poolformer/fpn_poolformer_m36_8xb4-40k_ade20k-512x512.py",
    "configs/knet/knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512.py",
    "configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py",
    "configs/twins/twins_svt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512.py"
]
for cfg_path in cfg_list:
    cfg = Config.fromfile(cfg_path)
    cfg_name = cfg_path.split("/")[-1]
    cfg.dump(
        os.path.join(
            "my_projects/configs",
            cfg_name
        )
    )

# cfg_list = []
# with open("my_projects/tests/model_select", 'r') as file:
#     for line in file:
#         if "config" in line:
#             cfg_path = line.split(":")[-1].strip()
#             if cfg_path not in cfg_list:
#                 cfg_list.append(cfg_path)

# for cfg_path in cfg_list:
#     print(cfg_path)
    
# for cfg_path in cfg_list:
#     cfg = Config.fromfile(cfg_path)
#     cfg.dump(
#         os.path.join(
#             "my_projects/configs",
#             cfg_path.split("/")[-1]
#         )
#     )
def get_param_count(cfg, trim_dict = True):
    
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    count_dict = parameter_count(model=model)
    
    if trim_dict:
        count_dict_ = {}
        count_dict, count_dict_ = count_dict_, count_dict
        for key, val in count_dict_.items():
            if "." in key:
                continue
            key_ = "model" if key == "" else key
            count_dict[key_] = count_dict_[key]   
    return count_dict

def print_count_dict(count_dict):
    for key, val in count_dict.items():
        print(f"{key} : {val}")
# # compare param count and table segformer_mit-b0:
# #   -> ade20k
# #   -> irl_vision
# #   -> hots_v1
# #   
# # compare mem/batch 
# # compare for diffent n_classes
# # compare resulted dumped cfgs  

# cfg_dump_path = "my_projects/tests/cfgs/"


# cfg_path_hots = "work_dirs/segformer_mit-b0_8xb2-160k_hots-v1-512x512/20240712_052908.py"
# cfg_hots = Config.fromfile(cfg_path_hots)
# param_count_dict = get_param_count(cfg=cfg_hots)
# print("hots: ")
# print_count_dict(param_count_dict)
# print("mem : 989")
# print("n_classes : 47\n")
# cfg_hots.dump(
#     os.path.join(
#         cfg_dump_path,
#         "hots.py"
#     )
# )


# cfg_path_irl = "work_dirs/segformer_mit-b0_8xb2-160k_irl_vision_sim-512x512/20240712_052441.py"
# cfg_irl = Config.fromfile(cfg_path_irl)
# param_count_dict = get_param_count(cfg=cfg_irl)
# print("irl_vision: ")
# print_count_dict(param_count_dict)
# print("mem : 1138")
# print("n_classes : 72\n")
# cfg_irl.dump(
#     os.path.join(
#         cfg_dump_path,
#         "irl_vision.py"
#     )
# )


# cfg_path_ade20k = "configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py"
# cfg_ade20k = Config.fromfile(cfg_path_ade20k)
# param_count_dict = get_param_count(cfg=cfg_ade20k)
# print("ade20k: ")
# print_count_dict(param_count_dict)
# print("mem : 2100")
# print("n_classes : 150\n")
# cfg_ade20k.dump(
#     os.path.join(
#         cfg_dump_path,
#         "ade20k.py"
#     )
# )

