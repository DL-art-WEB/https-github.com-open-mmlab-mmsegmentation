# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import torch
from multiprocessing import Pool, cpu_count
from copy import deepcopy
import time

import numpy as np
import torch
from mmengine import Config
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmseg.registry import MODELS

def fix_test_loader(cfg, dataset = "evaltest"):
    img_pth = os.path.join("img_dir", dataset)
    seg_map_path = os.path.join("ann_dir", dataset)
    cfg.test_dataloader.dataset.data_prefix["img_path"] = img_pth
    cfg.test_dataloader.dataset.data_prefix["seg_map_path"] = seg_map_path
    return cfg


def get_benchmark(
    config_path, checkpoint, 
    work_dir_path = None,
    repeat_times = 1,
    num_warmup = 5,
    pure_inf_time = 0,
    total_iters = 200,
    log_interval = 50,
    verbose = False
): 
    # torch.cuda.empty_cache()
    cfg = Config.fromfile(config_path)

    init_default_scope(cfg.get('default_scope', 'mmseg'))
    if verbose:
        print("scope init")
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    if work_dir_path is not None:
        mkdir_or_exist(osp.abspath(work_dir_path))
        json_file = osp.join(work_dir_path, f'fps_{timestamp}.json')
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(config_path))[0])
        mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, f'fps_{timestamp}.json')
    
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None
    if verbose:
        print("cfg pretrained set to none")
    benchmark_dict = dict(config=config_path, unit='img / s')
    overall_fps_list = []
    
    cfg.test_dataloader.batch_size = 1
    if verbose:
        print("batch_size to ")
    for time_index in range(repeat_times):
        if verbose:
            print(f'Run {time_index + 1}:')
        # build the dataloader
        cfg_test_dataloader = cfg.test_dataloader
        cfg_test_dataloader.sampler = dict(shuffle=True, type='InfiniteSampler')
        data_loader = Runner.build_dataloader(cfg_test_dataloader)
        if verbose:
            print("build dataload dome")
        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = MODELS.build(cfg.model)
        if verbose:
            print("build model done")
        if osp.exists(checkpoint):
            load_checkpoint(model, checkpoint, map_location='cpu')
        if verbose:
            print("loaded checkpoint")
        if torch.cuda.is_available():
            model = model.cuda()

        model = revert_sync_batchnorm(model)

        model.eval()

        # the first several iterations may be very slow so skip them
        

        # benchmark with 200 batches and take the average
        for i, data in enumerate(data_loader):
            data = model.data_preprocessor(data, True)
            inputs = data['inputs']
            data_samples = data['data_samples']
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                model(inputs, data_samples, mode='predict')

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    if verbose:
                        print(f'Done image [{i + 1:<3}/ {total_iters}], '
                            f'fps: {fps:.2f} img / s')

            if (i + 1) == total_iters:
                fps = (i + 1 - num_warmup) / pure_inf_time
                if verbose:
                    print(f'Overall fps: {fps:.2f} img / s\n')
                benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                overall_fps_list.append(fps)
                break
    
    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
    torch.cuda.empty_cache()
    return benchmark_dict

def trigger_visualization_hook(cfg, show_dir):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        visualization_hook["show"] = False
        if show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


    
def get_results_all_workdirs():
    results = []
    test_results_path = "test_results"
    work_dir_path = "work_dirs"
    one_test = False
    # unique false bc one_test True
    project_names = os.listdir(work_dir_path)
    # project_names = trimmed_projects(
    #     work_dir_path=work_dir_path, 
    #     test_results_path=test_results_path,
    #     unique=(not one_test)
    # )
    
    print(project_names)
    # project_names = ["bisenetv1_r18-d32-in1k-pre_1xb2-2k_hots-v1-512x512"]
    for project_name in project_names:
        if "-5k" not in project_name:
            continue
        print(f"evaluating project: {project_name}")
        project_path = os.path.join(work_dir_path, project_name)
        # checkpoint_names = trimmed_checkpoints(
        #     project_name=project_name,
        #     work_dir_path=work_dir_path,
        #     test_results_path=test_results_path
        # )
        if not os.path.exists(project_path):
            continue
        checkpoint_names = [    
            file for file in os.listdir(project_path) if ".pth" in file
        ]
        # TODO temp #################################
        if not checkpoint_names:
            continue
        best_checkpoint_names = [
            checkpoint_name for checkpoint_name in checkpoint_names
                if "best" in checkpoint_name
        ]
        if best_checkpoint_names:
            checkpoint_names = best_checkpoint_names
        else:
            max_check = checkpoint_names[0]
            max_iter = int(max_check.split("_")[-1].replace(".pth", ""))
            for checkpoint_name in checkpoint_names:
                iter = int(checkpoint_name.split("_")[-1].replace(".pth", ""))
                
                if iter > max_iter:
                    max_check = checkpoint_name
                    max_iter = iter
            checkpoint_names = [max_check]
        
        ####################################################
        for checkpoint_name in checkpoint_names:
            config_name = [file_name for file_name in os.listdir(project_path) if ".py" in file_name][0]
            config_path = os.path.join(project_path, config_name)
            test_work_dir_path = os.path.join(test_results_path, project_name)
            
            
            output_dir = os.path.join(test_work_dir_path, checkpoint_name, "out")
            show_dir = os.path.join(test_work_dir_path, checkpoint_name, "show")
            cfg = Config.fromfile(config_path)
            cfg = trigger_visualization_hook(cfg, show_dir)
            cfg.test_evaluator["output_dir"] = output_dir
            cfg.test_evaluator["keep_results"] = True
            torch.cuda.empty_cache()
            test_name = os.path.join(test_work_dir_path, checkpoint_name)
            cfg.work_dir = test_name
            checkpoint_path = os.path.join(project_path, checkpoint_name)
            cfg.load_from = checkpoint_path
            cfg.test_evaluator = dict(type="CustomIoUMetric")
            # cfg.test_evaluator = dict(type="IoUMetricFixed")
            
            # TEMP when using test eval merged dataset
            if "hots-v1" in config_name:
                cfg = fix_test_loader(cfg=cfg, dataset="evaltest")
            
            
            try:
                runner = Runner.from_cfg(cfg=cfg)
                
                metrics = runner.test()
                
            except Exception as exp:
                print(f"cfg: {test_name} test did not work")
                print(exp)
                continue
                
            try:
                benchmark_dict = get_benchmark(
                    config_path=config_path,
                    checkpoint=checkpoint_path,
                    verbose=False,
                    work_dir_path=test_name + "/bench"
                )    
            except Exception as exp:
                print(f"cfg: {test_name} benchmark did not work")
                print(exp)
                continue
            
            results.append(
                {
                    "name"          :       test_name,
                    "metric_dict"   :       metrics,
                    "benchmark_dict":       benchmark_dict
                }
            )
            
            
            
            torch.cuda.empty_cache()
            
            if one_test:
                exit()
    return results

def trimmed_projects(
    exclude = [], 
    unique = True, work_dir_path = "work_dirs",
    test_results_path = "test_results",
    training_iters = None
    ):
    def get_iters(project_name):
        train_set = project_name.split("_")[-2]
        iters = train_set.split("-")[-1]
        if 'k' in iters:
            iters = iters.replace('k', '000')
        return int(iters)
    
    project_names = os.listdir(work_dir_path)
    tested_project_names = os.listdir(test_results_path)
    project_names = [project_name for project_name in project_names if project_names not in exclude]
    if training_iters is not None:
        project_names = [project_name for project_name in project_names if training_iters == get_iters(project_name)]
    if unique:
        project_names = [project_name for project_name in project_names if project_name not in tested_project_names]
    # # TODO temp:
    # project_names = ["maskformer_r50-d32_1xb2-pre-ade20k-1k_hots-v1-512x512"]
    
    return project_names

def trimmed_checkpoints(
    project_name, work_dir_path = "work_dirs",
    test_results_path = "test_results", 
    exclude = [], unique = True
    ):
    project_path = os.path.join(work_dir_path, project_name)
    checkpoint_names = [file_name for file_name in os.listdir(project_path) if ".pth" in file_name]
    test_work_dir_path = os.path.join(test_results_path, project_name)
    if not os.path.exists(test_work_dir_path):
        return checkpoint_names
    tested_checkpoints = os.listdir(test_work_dir_path)
    if unique:
        checkpoint_names = [
            checkpoint_name for checkpoint_name in checkpoint_names
                if checkpoint_name not in tested_checkpoints
        ]
    return checkpoint_names

def print_result_list(result_list):
    for result in result_list:
        print(f"\nname: {result['name']}")
        print("metrics:")
        for key, val in result["metric_dict"].items():
            print(f"{key} : {val}")
        print("benchmark: ")
        for key, val in result["benchmark_dict"].items():
            print(f"{key} : {val}")
        print("-" * 80) 

def get_results_all_checkpoints(
    work_dir_path = "work_dirs",
    test_results_path = "test_results"
    ):
    results = []
    for project_name in os.listdir(work_dir_path):
        project_path = os.path.join(work_dir_path, project_name)
        cfg_name = [
            file for file in os.listdir(project_path) if ".py" in file
        ][0]
        cfg_path = os.path.join(project_path, cfg_name)
        for checkpoint_name in os.listdir(project_path):
            if ".pth" not in checkpoint_name:
                continue
            test_work_dir_path = os.path.join(test_results_path, project_name)
            output_dir = os.path.join(test_work_dir_path, checkpoint_name, "out")
            show_dir = os.path.join(test_work_dir_path, checkpoint_name, "show")
            cfg = Config.fromfile(cfg_path)
            cfg = trigger_visualization_hook(cfg, show_dir)
            cfg.test_evaluator["output_dir"] = output_dir
            cfg.test_evaluator["keep_results"] = True
            torch.cuda.empty_cache()
            test_name = os.path.join(test_work_dir_path, checkpoint_name)
            cfg.work_dir = test_name
            checkpoint_path = os.path.join(project_path, checkpoint_name)
            cfg.load_from = checkpoint_path
            cfg.test_evaluator = dict(type="CustomIoUMetric")
            # cfg.test_evaluator = dict(type="IoUMetricFixed")
            
            # TEMP when using test eval merged dataset
            if "hots-v1" in cfg_name:
                cfg = fix_test_loader(cfg=cfg, dataset="evaltest")
            
            
            try:
                runner = Runner.from_cfg(cfg=cfg)
                
                metrics = runner.test()
                
            except Exception as exp:
                print(f"cfg: {test_name} test did not work")
                print(exp)
                continue
                
            try:
                benchmark_dict = get_benchmark(
                    config_path=cfg_path,
                    checkpoint=checkpoint_path,
                    verbose=False,
                    work_dir_path=test_name + "/bench"
                )    
            except Exception as exp:
                print(f"cfg: {test_name} benchmark did not work")
                print(exp)
                continue
            
            results.append(
                {
                    "project_name"      :       project_name,
                    "checkpoint_name"   :       checkpoint_name,
                    "metric_dict"       :       metrics,
                    "benchmark_dict"    :       benchmark_dict
                }
            )
    print('#' * 80)
    for result in results:
        print(f"\nproject name : {result['project_name']}")
        print(f"checkpoint_name : {result['checkpoint_name']}")
        print(f"metric_dict: ")
        for key, val in result["metric_dict"].items():
            print(f"{key} : {val}")
        print(f"benchmark_dict: ")
        for key, val in result["benchmark_dict"].items():
            print(f"{key} : {val}")
    print('#' * 80)
    grouped = {}
    for result in results:
        if result["project_name"] not in grouped.keys():
            grouped[result["project_name"]] = []
        grouped[result["project_name"]].append(
            {
                "checkpoint_name"   :       result["checkpoint_name"],
                "metric_dict"       :       result["metrics"],
                "benchmark_dict"    :       result["benchmark_dict"]
            }
        )  
    print("GROUPED: ") 
    print('#' * 80)
    for project_name, checkpoints in grouped:
        print(f"project_name : {project_name}")
        for checkpoint in checkpoints:
            print(f"checkpoint_name : {checkpoint_name}")
            print("metrics:")
            for key, val in checkpoint["metric_dict"].items():
                print(f"{key} : {val}")
            print("benchmark: ")
            for key, val in checkpoint["benchmark_dict"].items():
                print(f"{key} : {val}") 
            print('-' * 80)
    print('#' * 80)
            
        

        
def main():
    get_results_all_checkpoints()

if __name__ == '__main__':
    main()
