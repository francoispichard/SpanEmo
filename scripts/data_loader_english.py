from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from transformers import logging as tr_logging
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='ekphrasis')
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')
tr_logging.set_verbosity_error()

import os
import sys
# Disable print
def disable_print():
    sys.stdout = open(os.devnull, 'w')

# Enable print
def enable_print():
    sys.stdout = sys.__stdout__


def twitter_preprocessor():
    disable_print()
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    enable_print()
    return preprocessor


class DataClass(Dataset):
    def __init__(self, max_length, filename, bert_tokeniser):
        self.max_length = max_length
        self.filename = filename
        self.data, self.labels = self.load_dataset()

        self.bert_tokeniser = bert_tokeniser 

        self.inputs, self.lengths, self.label_indices = self.process_data()

    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        try:
            df = pd.read_csv(self.filename, sep='\t')
            # ID and Tweet are the two columns in the DataFrame
            x_train = df.Tweet.apply(lambda row: str(row)).values
            # Dummy true labels. List of possible labels (11 columns): 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
            y_train = np.zeros((len(df), 11), dtype=int)
        except pd.errors.ParserError:
            print(f'Malformed input file: {self.filename}')
        return x_train, y_train

    def process_data(self):
        # Do not show messages from preprocessing module
        disable_print()
        preprocessor = twitter_preprocessor()

        segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
        label_names = ["anger", "anticipation", "disgust", "fear", "joy",
                       "love", "optimism", "hopeless", "sadness", "surprise", "trust"]

        inputs, lengths, label_indices = [], [], []
        # desc = "Preprocessing dataset {}...".format('')
        # for x in tqdm(self.data, desc=desc):
        for x in self.data:
            x = ' '.join(preprocessor(x))
            x = self.bert_tokeniser.encode_plus(segment_a,
                                                x,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                truncation=True)
            input_id = x['input_ids']
            input_length = len([i for i in x['attention_mask'] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)

            #label indices
            label_idxs = [self.bert_tokeniser.convert_ids_to_tokens(input_id).index(label_names[idx])
                             for idx, _ in enumerate(label_names)]
            label_indices.append(label_idxs)

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        enable_print()
        return inputs, data_length, label_indices

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        return inputs, labels, length, label_idxs

    def __len__(self):
        return len(self.inputs)
