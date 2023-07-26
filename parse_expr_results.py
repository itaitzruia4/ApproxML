import os
import numpy as np
import pandas as pd
import sys
import subprocess
import re

if len(sys.argv) < 2:
    print('Usage: python3 parse_expr_results.py <df_path>')
    exit(1)

df_path = sys.argv[1]

df = pd.read_csv(df_path)


def parse_jobs(jobs, rate):
    for idx, job in enumerate(jobs, start=1):
        # parse fitness
        file = f'job-{job}.out'
        file = file if os.path.exists(file) else os.path.join('results', file)
        if os.path.exists(file):
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # if 'best fitness:' in line.lower():
                    #     fitness = float(line.split(':')[-1])
                    #     if -1 <= fitness <= 0:
                    #         fitness *= 100_000
                    #     elif 0 < fitness <= 1:
                    #         fitness *= 2000
                    #     fitness_col = f'Unnamed: {str(int(float(rate) * 40 - 2))}'
                    #     df.at[idx, fitness_col] = fitness
                    if 'computations:' in line.lower():
                        n_comps = int(line.split(':')[-1])
                        computations_col = f'Unnamed: {str(int(float(rate) * 40 - 1))}'
                        df.at[idx, computations_col] = n_comps
            if float(rate) < 1:
                job_df = pd.read_csv(f'datasets/{job}.csv')
                fitness = job_df['fitness'].max()
                if -1 <= fitness <= 0:
                    fitness *= 100_000
                elif 0 < fitness <= 1:
                    fitness *= 2000
                fitness_col = f'Unnamed: {str(int(float(rate) * 40 - 2))}'
                df.at[idx, fitness_col] = fitness



cols = [str(round(x, 1)) for x in np.arange(0.2, 1, 0.2)]
rates2jobs = { 
    rate: [str(id) for id in df[rate].to_list()[1:31] if str(id).isnumeric()]
    for rate in cols
}

for rate, jobs in rates2jobs.items():
    parse_jobs(jobs, rate)

evo_jobs = [str(id) for id in df['evo'].to_list()[1:41] if str(id).isnumeric()]
parse_jobs(evo_jobs, rate=1.0)

if 'novelty-true' in df.columns:
    evo_opt_jobs = evo_jobs = [str(id) for id in df['novelty-true'].to_list()[1:31] if str(id).isnumeric()]
    parse_jobs(evo_jobs, rate=1.1)

# Write the updated DataFrame to a CSV file
df.to_csv(df_path, index=False)
