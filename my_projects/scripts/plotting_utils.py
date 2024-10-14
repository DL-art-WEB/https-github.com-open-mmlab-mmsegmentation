
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
    rcParams.update(
        param_dict
    )
    
def reset_params():
    rcParams.update(rcParamsDefault)
    
# def fix_metric_name(metric_name):
#     return metric_name.split(".")[0]

key_map = {
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
    if key in key_map.keys():
        return key_map[key]
    return key