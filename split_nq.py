import os
import csv
import json
import gzip
import argparse

import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict

title_s = "<title>"
title_e = "</title>"

def save(data_dir, data, data_type):
    file_path = os.path.join(data_dir, '{}.json'.format(data_type))
    with open(file_path, 'w') as f:
        print ("Saving {}".format(file_path))
        json.dump({'data': data}, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()
    for data_type in ['train', 'dev']:
        prepro_naturalquestions(args.data_dir, data_type)

def prepro_naturalquestions(data_dir, data_type):
    short_data_list = []
    for i in range(50 if data_type=='train' else 5):
        filename = os.path.join(data_dir, 'v1.0', data_type, 'nq-{}-{}.jsonl.gz'.format(data_type, str(i).zfill(2)))
        print ("Preprocessing {}".format(filename))
        with gzip.GzipFile(filename, 'r') as fin:
            json_bytes = fin.read()
            lines = json_bytes.decode('utf-8').split('\n')
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line)>0]
            for line in tqdm(lines):
                d = json.loads(line)
                question = d['question_text']
                document = [t['token'] for t in d['document_tokens']]
                answers = []
                for annotation in d['annotations']:
                    for short_annotation in annotation['short_answers']:
                        if short_annotation['end_token']-short_annotation['start_token']>5:
                            continue
                        answer = document[short_annotation['start_token']:short_annotation['end_token']]
                        answers.append(" ".join(answer))
                if len(answers)>0:
                    short_data_list.append({
                        'id': d['example_id'],
                        'question': question,
                        'answers': list(set(answers)),
                        'orig_doc_title': d['document_title']
                    })

    print (len(short_data_list))

    if data_type=='dev':
        save(data_dir, short_data_list, 'test')
    elif data_type=='train':
        np.random.seed(1995)
        indices = np.random.permutation(range(len(short_data_list)))
        short_data_list = [short_data_list[i] for i in indices]
        n_dev = 8757 # same number of dev data as Lee et al (ACL 2019)
        save(data_dir, short_data_list[:n_dev], 'dev')
        save(data_dir, short_data_list[n_dev:], 'train')
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
