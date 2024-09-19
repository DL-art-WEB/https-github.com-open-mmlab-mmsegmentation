import json 
import os
import argparse

KEYS_OF_INTEREST = {
    "performance"   :   [
        "mPr@50.0", 
        "mPr@60.0",
        "mPr@70.0",
        "mPr@80.0", 
        "mPr@90.0", 
        "mIoU"
    ],
    "benchmark"    :   [
        "average_fps",
        "fps_variance",
        "average_mem",
        "mem_variance"
    ],
    "compare"       :   [
        "mIoU",
        "average_fps",
        "average_mem"
    ]
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        '-dsp', 
        type=str,
        default=None,
        help='directory path of dataset results'
    )
    parser.add_argument(
        '--model_name', 
        '-mn',
        type=str,
        default=None,
        choices=[
            'bisenetv1', 
            'mask2former', 
            'maskformer', 
            'segformer',
            'segnext',
            None
        ],
        help='directory path of model results'
    )
    parser.add_argument(
        '--save_intermediate',
        '-si',
        action='store_true'
    )
    parser.add_argument(
        '--save_seperately',
        '-ss',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    return args

def save_dict_as_json(data_dict, dump_file_path):
    print(f"saving to : {dump_file_path}")
    with open(dump_file_path, 'w') as dump_file:
        json.dump(data_dict, dump_file, indent=4)
        
def load_json_file(json_file_path: str) -> dict:
    with open(json_file_path, 'r') as file:
        return json.load(file)
    
def is_acc_test(dir_name: str) -> bool:
    file_parts = dir_name.split("_")
    if len(file_parts) < 2:
        return False
    return file_parts[0].isnumeric() and file_parts[1].isnumeric()

def extract_json_file_name_from_dir(dir_path: str) -> str:
    for file_name in os.listdir(dir_path):
        if ".json" in file_name:
            return file_name
    return ""       

def get_data_dict(data_dict: dict, keys_of_interest: list) -> dict:
    return {
        key : val for key, val in data_dict.items()
            if key in keys_of_interest
    }

def get_data_dict_from_dir(dir_path: str, keys_of_interest: list) -> dict:
    json_file_name = extract_json_file_name_from_dir(
        dir_path=dir_path
    )
    data_dict = load_json_file(
        json_file_path=os.path.join(
            dir_path,
            json_file_name
        )
    )
    return get_data_dict(data_dict=data_dict, keys_of_interest=keys_of_interest)



def merg_dicts(dict_list: list) -> dict:
    new_dict = {}
    for dict_ in dict_list:
        for key, val in dict_.items():
            new_dict[key] = val
    return new_dict   
    

def generate_dict_model_results(model_results_path: str) -> dict:
    model_data_dict = {}
    for dir_name in os.listdir(model_results_path):
        if dir_name == "data":
            continue
        dir_path = os.path.join(model_results_path, dir_name)
        if os.path.isfile(dir_path):
            continue
        dict_ = {}
        if is_acc_test(dir_name=dir_name):
            dict_ = get_data_dict_from_dir(
                dir_path=dir_path,
                keys_of_interest=KEYS_OF_INTEREST['performance']
            )
        elif dir_name == "benchmark":
            dict_ = get_data_dict_from_dir(
                dir_path=dir_path,
                keys_of_interest=KEYS_OF_INTEREST['benchmark']
            )
        else:
            continue  
        model_data_dict = merg_dicts(
            dict_list=[
                model_data_dict,
                dict_
            ]
        )  
    return model_data_dict

def generate_dict_dataset_results(
    dataset_path: str, 
    save_intermediate: bool = False
) -> dict:
    dataset_results_dict = {}
    for model_dir in os.listdir(dataset_path):
        
        model_name = model_dir.split("_")[0]
        model_results_path = os.path.join(
            dataset_path,
            model_dir
        )
        if os.path.isfile(model_results_path):
            continue
        dataset_results_dict[model_name] = generate_dict_model_results(
            model_results_path=model_results_path
        )
        if save_intermediate:
            save_dict_as_json(
                data_dict=dataset_results_dict[model_name],
                dump_file_path=os.path.join(
                    model_results_path,
                    f"{model_name}_results.json"
                )
            )
    return dataset_results_dict


def get_model_results_path(model_name: str, dataset_path: str):
    for model_res_name in os.listdir(dataset_path):
        if model_name in model_res_name:
            return os.path.join(
                dataset_path, 
                model_res_name
            )


def main():
    args = parse_args()
    if not args.dataset_path:
        return
    if args.model_name:
        
        model_results_path = get_model_results_path(
            model_name=args.model_name,
            dataset_path=args.dataset_path
        )
        if model_results_path is None:
            print("no model res dir found ")
            return
        
        model_res_dict = generate_dict_model_results(
            model_results_path=model_results_path
        )
        # for key, val in model_res_dict.items():
        #     print(f"{key} : {val}")
    else:
        dataset_results_dict = generate_dict_dataset_results(
            dataset_path=args.dataset_path,
            save_intermediate=args.save_intermediate
        )
        _, dataset_name = os.path.split(args.dataset_path)
        if args.dataset_path[-1] == '/' or dataset_name == '':
            _, dataset_name = os.path.split(args.dataset_path[:-1])
        save_dict_as_json(
            data_dict=dataset_results_dict,
            dump_file_path=os.path.join(
                args.dataset_path,
                f"{dataset_name}_results.json"
            )
        )
        if args.save_seperately:
            performance_dict = {}
            bench_dict = {}
            compare_dict = {}
            for model_name, res_data in dataset_results_dict.items():
                performance_dict[model_name] = get_data_dict(
                    data_dict=res_data,
                    keys_of_interest=KEYS_OF_INTEREST['performance']
                )
                bench_dict[model_name] = get_data_dict(
                    data_dict=res_data,
                    keys_of_interest=KEYS_OF_INTEREST['benchmark']
                )
                compare_dict[model_name] = get_data_dict(
                    data_dict=res_data,
                    keys_of_interest=KEYS_OF_INTEREST['compare']
                )     
            save_dict_as_json(
                data_dict=performance_dict,
                dump_file_path=os.path.join(
                    args.dataset_path,
                    f"{dataset_name}_performance_results.json"
                )
            )   
            save_dict_as_json(
                data_dict=bench_dict,
                dump_file_path=os.path.join(
                    args.dataset_path,
                    f"{dataset_name}_benchmark_results.json"
                )
            ) 
            save_dict_as_json(
                data_dict=compare_dict,
                dump_file_path=os.path.join(
                    args.dataset_path,
                    f"{dataset_name}_compare_results.json"
                )
            ) 
                
            
            
        

if __name__ == '__main__':
    main()