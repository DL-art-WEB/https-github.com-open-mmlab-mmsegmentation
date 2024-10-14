
from matplotlib import rcParams, rcParamsDefault

double_col_readable = {
    'figure.figsize'    :   (12, 8),
    'legend.fontsize'   :   25,
    'axes.labelsize'    :   25,
    'axes.titlesize'    :   25,
    'xtick.labelsize'   :   25,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}

scenes_plot_global = {
    'figure.figsize'    :   (16, 8),
    'legend.fontsize'   :   30,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   28,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}

scenes_plot_global_per_scene_type = {
    'figure.figsize'    :   (12, 8),
    'legend.fontsize'   :   30,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   20,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}
scenes_plot_model = {
    'figure.figsize'    :   (18, 7),
    'legend.fontsize'   :   25,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   30,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}

clutter_plot = {
    'figure.figsize'    :   (12, 8),
    'legend.fontsize'   :   25,
    'axes.labelsize'    :   30,
    'axes.titlesize'    :   30,
    'xtick.labelsize'   :   25,
    'ytick.labelsize'   :   25,
    'lines.linewidth'   :   6
}
def set_params(param_dict=None):
    if param_dict is None:
        param_dict = double_col_readable
    param_dict_ = {}
    for key, val in param_dict.items():
        if key in rcParams.keys():
            param_dict_[key] = val
        else:
            print(f"key {key} not in rcParams")
    rcParams.update(
        param_dict_
    )
    
def reset_params():
    rcParams.update(rcParamsDefault)
    
# def fix_metric_name(metric_name):
#     return metric_name.split(".")[0]

KEY_MAP = {
    "mPr@50"      :    "$\mathregular{mPr_{50}}$",
    "mPr@60"      :    "$\mathregular{mPr_{60}}$",
    "mPr@70"      :    "$\mathregular{mPr_{70}}$",
    "mPr@80"      :    "$\mathregular{mPr_{80}}$",
    "mPr@90"      :    "$\mathregular{mPr_{90}}$",
    "mPr@50.0"    :    "$\mathregular{mPr_{50}}$",
    "mPr@60.0"    :    "$\mathregular{mPr_{60}}$",
    "mPr@70.0"    :    "$\mathregular{mPr_{70}}$",
    "mPr@80.0"    :    "$\mathregular{mPr_{80}}$",
    "mPr@90.0"    :    "$\mathregular{mPr_{90}}$"     
}

def map_key(key):
    if key in KEY_MAP.keys():
        return KEY_MAP[key]
    return key

MODEL_NAME_MAP = {
    "bisenet"       :       "BiSeNet",
    "bisenetv1"     :       "BiSeNet",
    "mask2former"   :       "Mask2Former",
    "maskformer"    :       "MaskFormer",
    "segformer"     :       "SegFormer",
    "segnext"       :       "SegNeXt",
}

def map_model_name(model_name):
    if model_name.lower() in MODEL_NAME_MAP.keys():
        return MODEL_NAME_MAP[model_name.lower()]
    return model_name

DATASET_NAME_MAP = {
    "hots"              :           "HOTS",
    "hots cat"          :           "HOTS-C",
    "irl vision"        :           "SOD",
    "irl vision cat"    :           "SOD-C",
    "arid20"            :           "ARID20"
}

def map_dataset_name(ds_name):
    ds_name_ = ds_name.lower()
    if ds_name_ in DATASET_NAME_MAP.keys():
        return DATASET_NAME_MAP[ds_name_]
    if "hots" in ds_name_:
        if "cat" or "c" in ds_name_:
            return DATASET_NAME_MAP["hots cat"]
        else:
            return DATASET_NAME_MAP["hots"]
    if "irl" in ds_name_:
        if "cat" or "c" in ds_name_:
            return DATASET_NAME_MAP["irl vision cat"]
        else:
            return DATASET_NAME_MAP["irl vision"]
    if "arid" in ds_name_:
        return DATASET_NAME_MAP["arid20"]
    return ds_name