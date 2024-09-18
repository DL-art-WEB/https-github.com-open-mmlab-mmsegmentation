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
        "save_file_name",
        type=str
    )
    args = parser.parse_args()
    if ".png" not in args.save_file_name:
        args.save_file_name = f"{args.save_file_name}.png"
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

def make_plot(clutter_data_dict, save_path):
    x_axis = [
        int(n_objects + 1) for n_objects in range(len(clutter_data_dict["mIoU"]))
    ]
    legend = []
    for metric_name, data in clutter_data_dict.items():
        plt.plot(x_axis, data, linewidth=3)
        legend.append(metric_name)
    plt.legend(legend)
    plt.xticks(x_axis)
    plt.ylabel("score (%)")
    plt.xlabel("n objects")
    plt.tight_layout()
    plt.savefig(fname=save_path, format='png')
    
def main():
    args = parse_args()
    clutter_path = args.clutter_path
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
            make_plot(
                clutter_data_dict=clutter_dict,
                save_path=os.path.join(clutter_path, args.save_file_name)
            )

if __name__ == '__main__':
    main()
