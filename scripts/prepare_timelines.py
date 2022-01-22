import pandas as pd
import os
import glob

import datetime
import joblib
import multiprocessing
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def select_columns(f_name, output_dir):
    # Basename without .jsonl extension
    f_basename = os.path.basename(f_name).split('.')[0]
    df = pd.read_json(f_name, lines=True)
    df = df[['id', 'text']].copy()
    df.columns = ['ID', 'Tweet']
    # Save selected columns
    df.to_csv(os.path.join(output_dir, f_basename + '.txt'), sep='\t', index=False)

def main():
    # Input data
    input_dir = os.path.join('/', 'drives', 'sde', 'francesco', 'longcovid', 'data', 'timelines', 'raw')
    
    f_names_list = glob.glob(os.path.join(input_dir, '*.jsonl'))

    # Create folder for output data
    time_fmt = '%Y-%m-%d_%H-%M-%S'
    timestamp = datetime.datetime.now()
    stamp_str = timestamp.strftime(time_fmt)
    output_dir = os.path.join('..', 'data', 'prepared_'+stamp_str)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    num_cpu_threads = max(multiprocessing.cpu_count()-1, 1)
    parallel = joblib.Parallel(n_jobs=num_cpu_threads) # 'loky' backend: single-host, process-based parallelism (used by default)
    logger.info(f'Using {num_cpu_threads} CPUs to extract relevant columns...')
    
    select_columns_delayed = joblib.delayed(select_columns)
    parallel(select_columns_delayed(f_name, output_dir) for f_name in tqdm(f_names_list, desc='Get relevant columns from raw JSON Line files', unit='file'))
    
    logger.info(f'Results (stored in .txt files) can be found here:\n {output_dir}')


if __name__ == '__main__':
    main()
