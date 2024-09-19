import subprocess
import os
# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         ''
#     )

DATASET_SOURCE_PATH = {
    "HOTS"          :       "my_projects/conversion_tests/finegrained_vs_general/selection_trained/hots_v1",
    "IRL_VISION"    :       "my_projects/conversion_tests/finegrained_vs_general/selection_trained/irl_vision"
}

# TODO maybe add irl2hots later
DATASET_SETUPS = [
    ("HOTS", "HOTS", "HOTS_CAT"),
    ("IRL_VISION", "IRL_VISION", "IRL_VISION_CAT"),
    ("IRL_VISION", "IRL_VISION", "HOTS_CAT"),
    ("HOTS", "HOTS", "IRL_VISION_CAT")
]

def run_conversion_test(
    model_dir_path: str,
    test_dataset: str,
    output_dataset: str,
    target_dataset: str,
    test_results_path: str,
    gen_cfg: bool = True,
    cfg_save_dir: str = "conv_cfg"
):
    subprocess.call(
        [
            "python",
            "my_projects/conversion_tests/scripts/conversion_test.py",
            "--model_dir_path",
            model_dir_path,
            "--test_dataset",
            test_dataset,
            "--output_dataset",
            output_dataset,
            "--target_dataset",
            target_dataset,
            "--test_results_path",
            test_results_path,
            "--gen_cfg",
            "--cfg_save_dir",
            cfg_save_dir  
        ]
    )
    
    
    

def main():
    results_dir_path = "my_projects/conversion_tests/finegrained_vs_general/results"
    for (test_dataset, output_dataset, target_dataset) in DATASET_SETUPS:
        dataset_src_path = DATASET_SOURCE_PATH[output_dataset]
        for model_dir in os.listdir(dataset_src_path):
            
            run_conversion_test(
                model_dir_path=os.path.join(
                    dataset_src_path,
                    model_dir
                ),
                test_dataset=test_dataset,
                output_dataset=output_dataset,
                target_dataset=target_dataset,
                test_results_path=os.path.join(
                    results_dir_path,
                    f"{output_dataset.lower()}2{target_dataset.lower()}"
                ),
                gen_cfg=True,
                cfg_save_dir="conv_cfg"
            )

if __name__ == '__main__':
    main()