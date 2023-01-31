import argparse
import json
import os
import warnings
import random
import numpy as np
from tqdm import tqdm

def get_img_id(annot):
    img_id = annot['img_fn']
    img_id = os.path.basename(img_id)
    img_id = img_id[:img_id.rfind('.')]
    return img_id

def get_text_data(annot, index):
    data = []
    event = annot['event']
    place = annot['place']
    metadata_dir = annot['metadata_fn']
    metadata = json.load(open(os.path.join(args.data_dir, metadata_dir)))

    img_id = get_img_id(annot)
    base_entry = {'event': event, 'img_id': img_id, 'img_fn': annot['img_fn'], 'index': index, 'place': place, 'names': metadata['names']}

    if annot['split'] == 'test':
        data.append(base_entry)
    else:
        for intent in annot['intent']:
            data.append({**base_entry, 'task_type': 'intent', 'labels': intent})
        for before in annot['before']:
            data.append({**base_entry, 'task_type': 'before', 'labels': before})
        for after in annot['after']:
            data.append({**base_entry, 'task_type': 'after', 'labels': after})

    return data


def get_eval_data(annot, index):
    data = []
    event = annot['event']
    place = annot['place']
    metadata_dir = annot['metadata_fn']
    metadata = json.load(open(os.path.join(args.data_dir, metadata_dir)))

    img_id = get_img_id(annot)
    base_entry = {'event': event, 'img_id': img_id, 'img_fn': annot['img_fn'], 'index': index, 'place': place, 'names': metadata['names']}

    if annot['split'] == 'test':
        data.append(base_entry)
    else:
        data.append({**base_entry, 'task_type': 'intent'})
        data.append({**base_entry, 'task_type': 'after'})
        data.append({**base_entry, 'task_type': 'before'})

    return data


def get_reference_data(annot):
    return [{
        'intent': annot.get('intent'),
        'before': annot.get('before'),
        'after': annot.get('after')
    }]


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='VCR image directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='VCG annotation directory'
                             'with "val_annots.json", "train_annots.json" and "test_annots.json"')
    parser.add_argument('--config', type=str, default=None,
                        help='path extractor config')
    parser.add_argument('--gpu_num', default=1, type=int,
                        help='number of GPUs in total')
    args = parser.parse_args()

    if args.config is None:
        args.config = 'config/extract_config.yaml'

    # load annotations
    origin_val_annots = json.load(open(os.path.join(args.annot_dir, 'val_annots.json')))

    # train_annots = json.load(open(os.path.join(args.annot_dir, 'train_annots.json')))
    # train_annots = json.load(open(os.path.join(args.annot_dir, 'train_annots_random_filtered.json')))
    train_annots = json.load(open(os.path.join(args.annot_dir, 'train_annots_freq.json')))
    val_annots = json.load(open(os.path.join(args.annot_dir, 'val_annots_freq_sim_cst_1.json')))
    test_annots = json.load(open(os.path.join(args.annot_dir, 'test_annots.json')))

    split_dict = {'train': train_annots, 'val_filtered': val_annots, 'test': test_annots, 'val': origin_val_annots}

    # make directory for splits
    # for split in split_dict.keys():
    #     path = os.path.join(args.output_dir, split)
    #     if not os.path.isdir(path):
    #         os.mkdir(path)

    # generate and save training data (event, task_type, etc.)
    # [{
    #       'event': sentence, 'img_id': image_id, 'img_fn': image_path,
    #       'index': event_index, 'task_type': task_type, 'labels': sentence
    #  }, ...]
    # print_segment_line('processing training data')
    for split, annots in split_dict.items():
        data = []
        for index, annot in enumerate(tqdm(annots)):
            data += get_text_data(annot=annot, index=index)
        # if split == 'val_filtered':
        #     split = split + '_filtered'
        json.dump(data, open(os.path.join(args.output_dir, split + '.json'), 'w'))

    # generate and save evaluation data (event, task_type, etc.)
    # [{
    #       'event': sentence, 'img_id': id, 'img_fn': image_path,
    #       'index': event_index, 'task_type': task_type
    #  }, ...]
    # print_segment_line('processing evaluation data')
    for split, annots in split_dict.items():
        data = []
        for index, annot in enumerate(tqdm(annots)):
            data += get_eval_data(annot=annot, index=index)
        # if split == 'val_filtered':
        #     split = split + '_filtered'
        json.dump(data, open(os.path.join(args.output_dir, split + '_eval.json'), 'w'))

    # generate and save reference data
    # [{
    #       'intent': [sentence1, sentence2, ...],
    #       'before': [sentence3, sentence4, ...],
    #       'after': [sentence5, sentence6, ...]
    #  }, ...]
    # print_segment_line('processing reference data')
    for split, annots in split_dict.items():
        if split != 'test':
            data = []
            for index, annot in enumerate(tqdm(annots)):
                data += get_reference_data(annot=annot)
            # if split == 'val_filtered':
            #     split = split + '_filtered'
            json.dump(data, open(os.path.join(args.output_dir, split + '_ref.json'), 'w'))