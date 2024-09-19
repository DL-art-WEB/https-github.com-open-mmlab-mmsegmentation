from matplotlib import pyplot as plt 
import numpy as np
import json 
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'clutter_path', 
        type=str,
        help='directory path of clutter experiment'
    )
    parser.add_argument(
        "save_dir_path",
        type=str
    )
    
    parser.add_argument(
        '--global_plot',
        '-gp',
        action='store_true'
    )
    parser.add_argument(
        '--per_model',
        '-pm',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    return args


def get_data_dict(json_path):
    data = None
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def reorganize_clutter_dict(clutter_data_dict):
    metrics_names = [
        "mPr@50.0", "mPr@60.0", "mPr@70.0", 
        "mPr@80.0", "mPr@90.0", "mIoU"
    ]
    cl_dict_new = {
        metric_name : np.zeros(len(clutter_data_dict))
            for metric_name in metrics_names
    }
    for n_objects, data_dict in clutter_data_dict.items():
        for metric_name, metric_val in data_dict.items():
            if metric_name not in metrics_names:
                continue
            cl_dict_new[metric_name][int(n_objects) - 1] = metric_val
    return cl_dict_new

def make_plot(
    clutter_data_dict,
    save_path, 
    xlabel="n objects",
    ylabel="score (%)"
):
    x_axis = [
        int(n_objects + 1) for n_objects in range(
            len(
                clutter_data_dict[list(clutter_data_dict.keys())[0]]
            )
        )
    ]
    y_axis = np.arange(0, 110, 10)
    legend = []
    for metric_name, data in clutter_data_dict.items():
        plt.plot(x_axis, data, linewidth=3)
        legend.append(metric_name)
    if ".png" not in save_path:
        save_path = f"{save_path}.png"
    plt.legend(legend, loc='lower left')
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(fname=save_path, format='png')
    plt.clf()

def get_clutter_dict(clutter_path: str):
    for file_name in os.listdir(clutter_path):
        if ".json" in file_name:
            clutter_dict = get_data_dict(
                json_path=os.path.join(
                    clutter_path,
                    file_name
                )
            )
            clutter_dict = reorganize_clutter_dict(
                clutter_data_dict=clutter_dict
            )
            return clutter_dict

def plot_clutter_per_model(
    clutter_dict, save_path
):
    
    make_plot(
        clutter_data_dict=clutter_dict,
        save_path=save_path
    )

def global_plot(args):
    dataset_path = args.clutter_path
    global_miou_dict = {}
    for model_dir in os.listdir(dataset_path):
        if model_dir == "data":
            continue
        model_path = os.path.join(dataset_path, model_dir)
        if os.path.isfile(model_path):
            continue
        clutter_path = os.path.join(
            model_path,
            "clutter"
        )
        clutter_dict = get_clutter_dict(
            clutter_path=clutter_path
        )
        model_name = model_dir.split("_")[0]
        if args.per_model:
            plot_clutter_per_model(
                clutter_dict=clutter_dict,
                save_path=os.path.join(
                    args.save_dir_path,
                    f"{model_name}_clutter.png"
                )
            )
        global_miou_dict[model_name] = clutter_dict["mIoU"]
    save_path = os.path.join(args.save_dir_path, f"global_clutter")
    make_plot(
        clutter_data_dict=global_miou_dict,
        save_path=save_path,
        xlabel="n_objects", ylabel="mIoU"
    )
 
def main():
    args = parse_args()
    if args.global_plot:
        global_plot(args=args)         
    else:
        clutter_dict = get_clutter_dict(clutter_path=args.clutter_path)
        plot_clutter_per_model(
            clutter_dict=clutter_dict,
            save_path=args.save_dir_path
        )

if __name__ == '__main__':
    main()
