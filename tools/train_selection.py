import subprocess
import os

def main():
    config_dir = "my_projects/configs/arid20_cat"
    for cfg_name in os.listdir(config_dir):
        if "mask" not in cfg_name:
            continue
        cfg_path = os.path.join(config_dir, cfg_name)
        print(f"cfg: {cfg_name}")
        call_list = [
            "python3",
            "tools/train.py",
            cfg_path 
        ]
        subprocess.call(call_list)
        



if __name__ == '__main__':
    main()


