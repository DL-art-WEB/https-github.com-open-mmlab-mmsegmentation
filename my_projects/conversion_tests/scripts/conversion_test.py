from mmengine import Config
import argparse
import os
import subprocess
from my_projects.conversion_tests.converters.conversion_dicts import(
    DATASET_CLASSES,
    DATASET_PALETTE
)
from tools.test_selection import run_performance_test


TEST_DATASET_TEMPLATE_PATHS = {
    "HOTS" :  "my_projects/conversion_tests/dataset_template_configs/hots_test_data.py",
    "IRL_VISION"   : "my_projects/conversion_tests/dataset_template_configs/irl_test_data.py"
}

TEST_DATASET_RESULTS_DIR_NAME = {
    "HOTS"          :       "hots_v1",
    "IRL_VISION"    :       "irl_vision"
}

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir_path',
        '-mdp',
        type=str,
        default="my_projects/zero_shot/models"
    )
    parser.add_argument(
        '--test_dataset',
        '-test_ds',
        type=str,
        default="HOTS",
        choices=["HOTS", "IRL_VISION"]
    )
    parser.add_argument(
        '--output_dataset',
        '-out_ds',
        type=str,
        default="HOTS",
        choices=["HOTS", "IRL_VISION", "ADE20K"]
    )
    parser.add_argument(
        '--target_dataset',
        '-tar_ds',
        type=str,
        default="HOTS_CAT",
        choices=["HOTS_CAT", "IRL_VISION_CAT"]
    )
    
    parser.add_argument(
        '--test_results_path',
        '-trp',
        type=str,
        default="work_dirs"
    )
    parser.add_argument(
        '--gen_cfg',
        '-g_cfg',
        action='store_true'
    )
    # parser.add_argument(
    #     '--cfg_name',
    #     type=str,
    #     default=""
    # )
    parser.add_argument(
        '--cfg_save_dir',
        '-cfg_dir',
        type=str,
        default="conv_cfg"
    )
    
    args = parser.parse_args()
    return args

def conversion_test_model(
    model_path: str, 
    cfg_path: str,
    test_results_path: str
):
    model_name = get_model_dir_name(model_path=model_path)
    
    checkpoint_path = get_checkpoint_path(model_path=model_path)
    work_dir_path = os.path.join(
        test_results_path,
        model_name
    )
    
    prediction_result_path = os.path.join(
        test_results_path,
        model_name,
        "pred_results"
    )
    
    show_dir_path = os.path.join(
        test_results_path,
        model_name,
        "show"
    )
    run_performance_test(
        cfg_path=cfg_path, 
        checkpoint_path=checkpoint_path,
        work_dir_path=work_dir_path, 
        prediction_result_path=prediction_result_path,
        show_dir_path=show_dir_path
    )


def get_checkpoint_path(model_path: str) -> str:
    return os.path.join(
        model_path,
        "weights.pth"
    )
    
def get_base_cfg_path(model_path: str) -> str:
    return os.path.join(
        model_path,
        f"{get_model_dir_name(model_path=model_path)}.py"
    )
    
def get_model_dir_name(model_path: str):
    idx_ = -2 if model_path[-1] == '/' else -1
    return model_path.split("/")[idx_]        

def gen_conversion_cfg(
    base_cfg_path: str,
    args
) -> str:    
    cfg = Config.fromfile(base_cfg_path)
    ds_template_cfg = Config.fromfile(
        TEST_DATASET_TEMPLATE_PATHS[args.test_dataset]
    )
    cfg.merge_from_dict(ds_template_cfg.to_dict())
    cfg.merge_from_dict(
        options=dict(
            test_evaluator = dict(
                type="CustomIoUMetricZeroShot",
                test_dataset=args.test_dataset,
                output_dataset=args.output_dataset,
                target_dataset=args.target_dataset
            )
        )
    )
    
    cfg.merge_from_dict(
        options=dict(
            visualizer = dict(
                name='visualizer',
                type='SegLocalVisualizerZeroShot',
                vis_backends=[
                    dict(type='LocalVisBackend'),
                ],
                test_dataset=args.test_dataset,
                output_dataset=args.output_dataset,
                target_dataset=args.target_dataset
            )
        )
    )
    return cfg

def dump_conversion_cfg(
    cfg: Config,
    save_path: str
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cfg.dump(save_path)    

def gen_conversion_cfg_name(args) -> str:
    model_name = get_model_dir_name(model_path=args.model_dir_path)
    model_name = model_name.split("_")[0]
    test = args.test_dataset.lower()
    out = args.output_dataset.lower()
    target = args.target_dataset.lower()
    return f"{model_name}_{test}_{out}_{target}.py"
     
   
def gen_conversion_cfg_path(args):
    return os.path.join(
        args.model_dir_path,
        args.cfg_save_dir,
        conv_cfg_name = gen_conversion_cfg_name(args=args)
    )
    

def main():
    
    args = arg_parse()
    conv_cfg_path = get_base_cfg_path(args=args)
    
    if args.gen_cfg or not os.path.exists(conv_cfg_path):
        cfg = gen_conversion_cfg(
            base_cfg_path=get_base_cfg_path(args.model_dir_path),
            args=args
        )
        dump_conversion_cfg(cfg=cfg, save_path=conv_cfg_path)
    
    conversion_test_model(
        model_path=args.model_dir_path,
        cfg_path=conv_cfg_path,
        test_results_path=args.test_results_path
    )
    
    
        
if __name__ == '__main__':
    main()
    