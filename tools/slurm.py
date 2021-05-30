import sys
import os
import json

def makefile(jobname,index,runcommand,condaenv):
    name = jobname + '-' + str(index)
    if not os.path.exists("jobs/"): 
        os.mkdir("jobs")
        os.mkdir("jobs/outputs")
    
    config = None
    with open(f"config/slurm-config.json") as cfg:
        config = json.load(cfg)

    with open(f"jobs/{name}.job",'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f'#SBATCH --job-name={name}\n')
        f.write(f'#SBATCH --output=outputs/{name}.out\n')
        f.write(f'#SBATCH --cpus-per-task={config["cpus-per-task"]}\n')
        f.write(f'#SBATCH --time={config["hours"]}:00:00\n')
        f.write('#SBATCH --gres=gpu\n')
        f.write('#SBATCH --mail-type=BEGIN,END,FAIL\n')
        f.write(f'#SBATCH --partition={config["partition"]}\n')
        f.write('\n')
        f.write('module load Anaconda3/2019.10\n')
        f.write('module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4\n')
        f.write('. $(conda info --base)/etc/profile.d/conda.sh\n')
        f.write(f'cd ..\n')
        f.write(f'conda activate {condaenv}\n')
        f.write(runcommand)
