import os
import sys
import glob
from tqdm import tqdm
import joblib
import multiprocessing

import math
import pandas as pd
import numpy as np

import time
import datetime
import argparse
from fastprogress.fastprogress import progress_bar, master_bar, format_time
import torch
from torch.utils.data import DataLoader
from data_loader_english import DataClass
from model import SpanEmo

from transformers import BertTokenizer
bert_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Disable print
def disable_print():
    sys.stdout = open(os.devnull, 'w')

# Enable print
def enable_print():
    sys.stdout = sys.__stdout__

class ArgParseDefault(argparse.ArgumentParser):
    """Simple wrapper which shows defaults in help"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

class PredictOnUnlabelled(object):
    def __init__(self, model, unlabelled_data_loader, checkpoint_path):
        self.model = model
        self.data_loader = unlabelled_data_loader
        self.checkpoint_path = checkpoint_path
    
    def predict(self, device='cuda:0', pbar=None):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.to(device).load_state_dict(checkpoint)
        self.model.eval()
        preds_array = np.empty((0, 11))

        with torch.no_grad():
            for step, batch in enumerate(progress_bar(self.data_loader, parent=pbar, leave=(pbar is not None))):
                # We are not interested in storing the loss and the dummy labels
                _, num_rows, y_pred, _ = self.model(batch, device)
                # y_pred is a two-dimensional Numpy array (on CPU)
                preds_array = np.concatenate((preds_array, y_pred), axis=0)

        return preds_array

def preprocess_file(input_filepath):
    # Define Dataloader
    unlabelled_dataset = DataClass(args.max_length, input_filepath, bert_tokeniser)
    unlabelled_data_loader = DataLoader(unlabelled_dataset, batch_size=args.batch_size, shuffle=False)
    return [input_filepath, unlabelled_data_loader]

def main(args):
    # CPU/GPU config
    def check_device(device):
        if str(device) == 'cuda:0':
            print("Currently using GPU: {}".format(device))
            np.random.seed(int(args.seed))
            torch.cuda.manual_seed_all(int(args.seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Use CPUs for preprocessing
    device = 'cpu'
    check_device(device)

    # Input data
    input_folder = os.path.join('..', 'data', f'prepared_{args.prepare_name}')
    input_files_list = glob.glob(os.path.join(input_folder, '*.txt'))

    # Create folder for output data
    time_fmt = '%Y-%m-%d_%H-%M-%S'
    run_time = datetime.datetime.now()
    run_time = run_time.strftime(time_fmt)
    output_folder = os.path.join('..', 'data', f'processed_tweets_{run_time}')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Column names
    emotion_labels = ["anger", "anticipation", "disgust", "fear", "joy",
                      "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
    # Load instance of SpanEmo class
    model = SpanEmo(lang='English')
    # Checkpoint of trained model
    checkpoint_path = os.path.join('..', 'models', args.model_name)
    
    num_worker_processes = multiprocessing.cpu_count()-10

    print('About to start processing...')
    s_time = datetime.datetime.now()
    num_files = len(input_files_list)
    chunk_size = 150
    num_chunks = math.ceil(num_files/chunk_size)
    for i in range(num_chunks):
        # Use CPUs for preprocessing
        device = torch.device('cpu')
        check_device(device)
        current_files_list = input_files_list[(chunk_size*i):(chunk_size*(i+1))]
        # Creating a pool object (indicate number of worker_processes to use
        pool = multiprocessing.Pool(num_worker_processes)
        # Map input_files_list to target function preprocess_file
        fpaths_dataloaders = list(tqdm(pool.imap(preprocess_file, current_files_list), total=len(current_files_list)))
        # Prevent any more tasks from being submitted to the pool
        pool.close()
        # Stop execution of current program until process is complete (wait for the worker processes to exit)
        pool.join()
        
        # Use GPU for processing
        device = torch.device('cuda:0') 
        check_device(device)

        for filepath_and_dataloader in fpaths_dataloaders:
            input_filepath = filepath_and_dataloader[0]
            data_loader = filepath_and_dataloader[1]
            # Run model
            learn = PredictOnUnlabelled(model, data_loader, checkpoint_path) 
            # Predict labels (y_pred is a Numpy array)
            y_pred = learn.predict(device=device)
            # Create a DataFrame for the prediction results
            emotions_df = pd.DataFrame(y_pred, columns=emotion_labels)
            # Summarize label predictions in a single column
            emotions_df['label'] = emotions_df[emotion_labels].apply(lambda row: row[row==1].index.values if len(row[row==1].index) > 0 else ['neutral'], axis=1) 
            # Drop original label columns
            emotions_df.drop(emotion_labels, axis=1, inplace=True)
            # Load input data and add new columns to the associated DataFrame
            id_text_df = pd.read_csv(input_filepath, sep='\t')
            # Concatenate both DataFrames
            id_text_df = pd.concat([id_text_df, emotions_df], axis=1)
            # Save data
            output_fname = os.path.basename(input_filepath).split('.')[0] + '.parquet'
            id_text_df.to_parquet(os.path.join(output_folder, output_fname))

        num_remaining_files = num_files - (i+1)*chunk_size
        current_time = datetime.datetime.now()
        elapsed_time = ((current_time - s_time).seconds) / 60
        print(f'Number of remaining files: {num_remaining_files:,}. Time elapsed since beginning of processing: {elapsed_time:.2f} min')
    
    e_time = datetime.datetime.now()
    processing_time = (e_time - s_time).seconds / 60

    print(f'Processing completed in {processing_time:.2f} min. Label predictions can be found here: \n {output_folder}')

def parse_args():
    parser = ArgParseDefault()
    parser.add_argument('--model-name', dest='model_name', type=str, required=True, help='Name of the checkpoint of a trained model. Model checkpoints are stored in the models/ folder')
    parser.add_argument('--prepare-name', dest='prepare_name', type=str, required=True, help='Name of the run (YYYY-MM-DD_HH-mm-SS) corresponding to data preparation. The input data are in a data/ subfolder with that timestamp. Input data are txt files with two columns ("ID", "Tweet")')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-length', dest='max_length', type=int, default=128, help='Maximum text length')
    parser.add_argument('--seed', type=int, default=0, help='The cuda manual seed should be set if you want to have reproducible results when using random generation on the GPU.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
