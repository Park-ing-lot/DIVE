from copy import deepcopy
import json
import os
import pickle
import re
import cv2
import numpy as np
import random

from torch.utils.data import Dataset

from src.utils import TaskType

"""
The dataset return format: a dictionary
{
    'task_type': ...        # TaskType
    'image_features': ...   # list[ndarray], optional
    'event': ...            # str, optional
    'labels': ...           # str, optional
    'index': ...            # int, optional, the index of reference data
    other task specific items...
}
"""


class COCODataset(Dataset):
    def __init__(self, data_dir, image_dir=None, split='train', eval_mode=False, use_image=True):
        """
        :param eval_mode: bool, if set to true, "xxx_eval.json" will be loaded, every image will only appear one time
        """
        self._use_image = use_image
        self._data_dir = data_dir
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split

        file_name = split + ('_eval.json' if eval_mode else '.json')
        self._dataset = json.load(open(os.path.join(data_dir, file_name), 'r'))

    def __getitem__(self, index):
        raw_data = self._dataset[index]
        output = {**raw_data}

        if self._use_image:
            image_dir = os.path.join(self._image_dir, self._split, str(raw_data['img_id']) + '.pkl')
            image_data = pickle.load(open(image_dir, 'rb'))
            output['image_features'] = np.concatenate([
                image_data['image_features'],
                image_data['boxes']
            ], axis=1).astype(np.float32)

            if 'mrm_labels' in image_data:
                output['mrm_labels'] = image_data['mrm_labels']

        return output

    def __len__(self):
        return len(self._dataset)


class VCGDataset(COCODataset):
    def __init__(
            self,
            data_dir,
            image_dir=None,
            split='train',
            eval_mode=False,
            use_image=True,
            use_event=True,
            pretrain=False,
    ):
        super(VCGDataset, self).__init__(
            data_dir=data_dir,
            image_dir=image_dir,
            split=split,
            eval_mode=eval_mode,
            use_image=use_image
        )
        self._use_event = use_event
        self._pretrain = pretrain

    def __getitem__(self, item):
        output = super(VCGDataset, self).__getitem__(item)
        
        if not self._use_event:
            output['event'] = output['event'].split()[0]  # only show the target person
        if self._pretrain:
            output['labels'] = output['event']
            del output['event']
            output['task_type'] = TaskType.CAPTION

        return output


class SBUDataset(COCODataset):
    def __init__(self, data_dir, image_dir=None, split='train', use_image=True):
        super(SBUDataset, self).__init__(
            data_dir=data_dir,
            image_dir=image_dir,
            split=split,
            eval_mode=False,
            use_image=use_image
        )

    def __getitem__(self, item):
        output = super(SBUDataset, self).__getitem__(item)
        output['task_type'] = TaskType.CAPTION
        output['labels'] = output['labels'].strip()
        return output


class CCDataset(SBUDataset):
    pass


class VGDataset(Dataset):
    def __init__(self, data_dir, image_dir=None, split='train'):
        self._data_dir = data_dir
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split

        self._dataset = json.load(open(os.path.join(data_dir, split + '.json'), 'r'))
        self._region_dataset = json.load(open(os.path.join(data_dir, split + '_region.json'), 'r'))

    def __len__(self):
        return len(self._region_dataset)

    def __getitem__(self, index):
        region_data = self._region_dataset[index]
        img_id = region_data['img_id']
        region_id = region_data['region_id']
        raw_data = self._dataset[str(img_id)]
        output = {**raw_data}

        image_dir = os.path.join(self._image_dir, self._split, str(raw_data['img_id']) + '.pkl')
        image_data = pickle.load(open(image_dir, 'rb'))

        region_index = image_data['region_ids'].index(region_id)
        region_feature = np.concatenate([
            image_data['region_features'][region_index],
            image_data['region_boxes'][region_index]
        ], axis=0)

        image_feature = np.concatenate([
            image_data['image_feature'],
            image_data['image_box']
        ], axis=0)

        object_features = np.concatenate([
            image_data['object_features'],
            image_data['object_boxes']
        ], axis=1)

        output['image_features'] = np.concatenate([
            image_feature[np.newaxis, :],
            object_features,
            region_feature[np.newaxis, :]
        ], axis=0)

        output['mrm_labels'] = np.concatenate([
            image_data['image_score'][np.newaxis, :],
            image_data['object_scores'],
            image_data['region_scores'][region_index: region_index+1]
        ], axis=0)

        output['object_ids'] = image_data['object_ids']
        output['task_type'] = TaskType.REGION_CAPTION
        output['labels'] = region_data['description']

        return output


class ReasonDataset(Dataset):
    def __init__(self, data_dir, image_dir=None, split='train', eval_mode=False, use_image=True, use_event=True):
        """
        :param eval_mode: bool, if set to true, "xxx_eval.json" will be loaded, every image will only appear one time
        """
        self._use_image = use_image
        self._use_event = use_event
        self._data_dir = data_dir
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split

        file_name = 'reason_' + split + ('_eval.json' if eval_mode else '.json')
        self._dataset = json.load(open(os.path.join(data_dir, file_name), 'r'))

    def __getitem__(self, index):
        raw_data = self._dataset[index]
        output = {**raw_data}
        
        if not self._use_event:
            output['event'] = ''
            
        if self._use_image:
            try:
                image_dir = os.path.join(self._image_dir, self._split, str(raw_data['img_id']) + '.pkl')
                image_data = pickle.load(open(image_dir, 'rb'))
            except FileNotFoundError:
                return None

            output['image_features'] = np.concatenate([
                image_data['image_features'],
                image_data['boxes']
            ], axis=1).astype(np.float32)

            if 'mrm_labels' in image_data:
                output['mrm_labels'] = image_data['mrm_labels']

        output['dataset_index'] = index

        return output

    def get_raw_data(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


class COCODataset_gt(Dataset):
    def __init__(self, data_dir, image_dir='../vcr/visualcomet/features/', split='train', eval_mode=False, use_image=True, get_negative_sample=False):
        """
        :param eval_mode: bool, if set to true, "xxx_eval.json" will be loaded, every image will only appear one time
        """
        self._use_image = use_image
        self._data_dir = data_dir
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split
        # self._vcr_dir = '/home/vcr/vcr1images/'

        file_name = split + ('_eval.json' if eval_mode else '.json')
        # file_name = split + ('_eval.json' if eval_mode else '_eval_label.json')
        self._dataset = json.load(open(os.path.join(data_dir, file_name), 'r'))

        if split=='train' and get_negative_sample:
            inference_count = {}
            for i, annot in enumerate(self._dataset):
                inference = annot['labels']
                inference = self.use_same_id(inference)
                if inference in inference_count.keys():
                    inference_count[inference] += 1
                else:
                    inference_count[inference] = 1

            tmp = {}
            for i, data in enumerate(self._dataset):
                inference = self.use_same_id(data['labels'])
                if inference not in tmp.keys():
                    tmp[inference] = [i]
                else:
                    tmp[inference].append(i)
            
            negative_dict = {}
            for key, value in tmp.items():
                if not len(value) == 1:
                    negative_dict[key] = value
            
            ### Including generic inferences for CRL ###
            # idx_dict = {}
            # for i, data in enumerate(self._dataset):
            #     index = data['index']
            #     if index not in idx_dict.keys():
            #         idx_dict[index] = [i]
            #     else:
            #         idx_dict[index].append(i)
    
            # self.others_dict = {}
            
            # for key, value in negative_dict.items():
            #     for i in value:
            #         if i in self.others_dict.keys(): continue # 이미 있으면 패스. 완성된거니까
            #         index = self._dataset[i]['index']
            #         tmp = deepcopy(idx_dict[index])
            #         if len(tmp) != 1:
            #             if i in tmp:
            #                 tmp.remove(i)
            #         self.others_dict[i] = tmp

            # self.negative_dict = negative_dict

            ### Excluding generic inferences for CRL ###
            idx_dict = {}
            for i, data in enumerate(self._dataset):
                index = data['index']
                if inference_count[self.use_same_id(data['labels'])] != 1: continue
                if index not in idx_dict.keys():
                    idx_dict[index] = [i]
                else:
                    idx_dict[index].append(i)
    
            self.others_dict = {}
            
            new_negative = {}
            for key, value in negative_dict.items():
                ndit_tmp = []
                for i in value:
                    if i in self.others_dict.keys(): continue # 이미 있으면 패스. 완성된거니까
                    index = self._dataset[i]['index']
                    if index not in idx_dict.keys(): continue # 하나의 샘플에 대해 unique가 없다면 패스
                    tmp = deepcopy(idx_dict[index])
                    if len(tmp) != 1:
                        if i in tmp:
                            tmp.remove(i)
                    self.others_dict[i] = tmp
                    ndit_tmp.append(i)
                if ndit_tmp != []:
                    new_negative[key] = ndit_tmp

            self.negative_dict = new_negative

                
        else:
            self.negative_dict = {}
            self.others_dict = {}
    
    def use_same_id(self, sent):
        r_sent = sent.replace("'", " '")
        r_sent = ' '.join([g if not g.isdigit() else '1' for g in r_sent.split()]).strip()
        r_sent = r_sent.replace(" '", "'")
        return r_sent

    def __getitem__(self, index):
        raw_data = self._dataset[index]
        output = {**raw_data}
        # print(raw_data)
        '''
        {'event': '1 is trying to wake up on Christmas morning', 
        'img_id': '8PB5sU_QcUc@42', 
        'img_fn': 'movieclips_Carrie_Pilby/8PB5sU_QcUc@42.jpg', 
        'index': 7936, 
        'place': 'in a living room', 
        'task_type': 'before', 
        'labels': "have a poor night's sleep"}
        '''
        if 'labels' in output.keys():
            tmp_inference = self.use_same_id(output['labels'])
        else:
            tmp_inference = None
        negative_index = index
        if tmp_inference in self.negative_dict.keys() and len(self.negative_dict[tmp_inference]) != 1: ###
            while index == negative_index:
                negative_index = random.sample(self.negative_dict[tmp_inference], 1)[0]
            negative_index = random.sample(self.others_dict[negative_index], 1)[0]
            
            raw_negative_sample = self._dataset[negative_index]
            for key, value in raw_negative_sample.items():
                output[f'negative_{key}'] = value

            # if len(self.negative_dict[tmp_inference]) > 1: ###
            #     negative_index = random.sample(self.negative_dict[tmp_inference], 2) ###
            # else:
            #     negative_index = self.negative_dict[tmp_inference]
            # for i, ni in enumerate(negative_index):
            #     raw_negative_sample = self._dataset[ni]
            #     if i == 0:
            #         for key, value in raw_negative_sample.items():
            #             output[f'negative_{key}'] = [value]
            #     else:
            #         for key, value in raw_negative_sample.items():
            #             output[f'negative_{key}'].append(value)
            

        else:
            for key, value in raw_data.items():
                output[f'negative_{key}'] = None

        if self._use_image:
            image_dir = os.path.join(self._image_dir, str(raw_data['img_id']) + '.pkl')
            # dict_keys(['image_features', 'object_features'])
            image_data = pickle.load(open(image_dir, 'rb'))

            output['image_features'] = np.row_stack((image_data['image_features'], image_data['object_features']))
            if 'negative_labels' in output.keys() and output['negative_labels'] is not None: ###
                image_dir = os.path.join(self._image_dir, str(raw_negative_sample['img_id']) + '.pkl')
                image_data = pickle.load(open(image_dir, 'rb'))
                output['negative_image_features'] = np.row_stack((image_data['image_features'], image_data['object_features']))

                # for i, ni in enumerate(negative_index):
                #     raw_negative_sample = self._dataset[ni]
                #     image_dir = os.path.join(self._image_dir, str(raw_negative_sample['img_id']) + '.pkl')
                #     image_data = pickle.load(open(image_dir, 'rb'))
                #     if i == 0:
                #         output['negative_image_features'] = [np.row_stack((image_data['image_features'], image_data['object_features']))]
                #     else:
                #         output['negative_image_features'].append(np.row_stack((image_data['image_features'], image_data['object_features'])))
            else:
                output['negative_image_features'] = None

            if 'mrm_labels' in image_data:
                output['mrm_labels'] = image_data['mrm_labels']

        # del output['names']

        return output

    def __len__(self):
        return len(self._dataset)


class VCGDataset_gt(COCODataset_gt):
    def __init__(
            self,
            data_dir,
            image_dir='../vcr/visualcomet/features/',
            split='train',
            eval_mode=False,
            use_image=True,
            use_event=True,
            use_place=True,
            use_others=False,
            pretrain=False,
            get_negative_sample=False
    ):
        super(VCGDataset_gt, self).__init__(
            data_dir=data_dir,
            image_dir=image_dir,
            split=split,
            eval_mode=eval_mode,
            use_image=use_image,
            get_negative_sample=get_negative_sample
        )
        self._use_event = use_event
        self._use_place = use_place
        self._pretrain = pretrain
        self.use_others = use_others

    def __getitem__(self, item):
        output = super(VCGDataset_gt, self).__getitem__(item)
        
        if not self.use_others:
            if 'other_responses' in output.keys():
                del output['other_reponses']
        if not self._use_place:
            del output['place']
        if not self._use_event:
            output['event'] = output['event'].split()[0]  # only show the target person
        else:
            if self._pretrain:
                label = output['labels']
                target_person = output['event'].split()[0]
                if not target_person.isdigit():
                    target_person = '1'

                if output['task_type'] == 'before':
                    label = target_person + ' needed to ' + label
                elif output['task_type'] == 'intent':
                    label = target_person + ' wanted to ' + label
                else:
                    label = target_person + ' will most likely ' + label

                output['labels'] = output['event']
                output['event'] = label
                output['task_type'] = TaskType.CAPTION

            person_in_event = re.findall("\d+", output['event']) # max 42
            tmp = []
            for e in output['event'].split():
                if e in person_in_event:
                    if int(e) <= 24:
                        e = f'<person{e}> '
                tmp.append(e)
            
            new_event = ' '.join(tmp)
            output['event'] = new_event

            if output['negative_event'] is not None: ###
                person_in_event = re.findall("\d+", output['negative_event']) # max 42
                tmp = []
                for e in output['negative_event'].split():
                    if e in person_in_event:
                        if int(e) <= 24:
                            e = f'<person{e}> '
                    tmp.append(e)
                
                new_event = ' '.join(tmp) 
                output['negative_event'] = new_event 

                # for i in range(len(output['negative_event'])):
                #     person_in_event = re.findall("\d+", output['negative_event'][i]) # max 42
                #     tmp = []
                #     for e in output['negative_event'][i].split():
                #         if e in person_in_event:
                #             if int(e) <= 24:
                #                 e = f'<person{e}> '
                #         tmp.append(e)
                    
                #     new_event = ' '.join(tmp)
                #     output['negative_event'][i] = new_event

        return output


class COCODataset_gt_retrieval(Dataset):
    def __init__(self, data_dir, 
                 image_dir='../vcr/visualcomet/features/', 
                 split='val', 
                 generation=None, 
                 eval_mode=False, 
                 use_image=True, 
                 index=0, 
                 num_candidates=1000,
                 val_data=None,
                 ):
        """
        :param eval_mode: bool, if set to true, "xxx_eval.json" will be loaded, every image will only appear one time
        """
        self._use_image = use_image
        self._data_dir = data_dir 
        self._generations = generation # must be the generated data 
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split
        # self._vcr_dir = '/home/vcr/vcr1images/'
        self.index = index

        file_name = split + ('_eval.json' if eval_mode else '.json')
        # file_name = split + ('_eval.json' if eval_mode else '_eval_label.json')
        # self._val = self._generations
        val = json.load(open(os.path.join(data_dir, 'val_annots.json'), 'r'))
        self._val = val_data
        '''
        a = json.load(open('/home/user16/HT/KM-BART-ACL/cf_generated/sample_1'))
        a[0]
        {'index': 0, 'task_type': 'intent', 'generations': ['read the note later']}
        '''
        
        dataset = []
        ### image retrieval
        base = deepcopy(self._generations[self.index])
        meta = val[base['index']]
        base['labels'] = base['generations'][0]
        del base['generations']
        base['event'] = meta['event']
        base['place'] = meta['place']
        base['img_id'] = meta['img_fn'].split('/')[-1][:-4]
        base['img_fn'] = meta['img_fn']

        ### text retrieval
        # base = deepcopy(self._val[self.index])

        dataset.append(base)
        # index 0 is always a label.
        random_idx = random.sample([i for i in range(len(self._val))], num_candidates)
        for i in random_idx:
            data = self._val[i]
            tmp = deepcopy(base)
            # tmp['labels'] = data['labels']
            tmp['event'] = data['event']
            tmp['place'] = data['place']
            tmp['img_id'] = data['img_fn'].split('/')[-1][:-4]
            tmp['img_fn'] = data['img_fn']
            dataset.append(tmp)

        # for i, data in enumerate(self._val):
        #     tmp = deepcopy(base)
        #     tmp['event'] = data['event']
        #     tmp['place'] = data['place']
        #     tmp['img_id'] = data['img_fn'].split('/')[-1][:-4]
        #     tmp['img_fn'] = data['img_fn']
        #     dataset.append(tmp)
            
        self._dataset = dataset

        # self.image_dir = os.path.join(self._image_dir, str(base['img_id']) + '.pkl')
        # self.image_data = self.image_data = pickle.load(open(self.image_dir, 'rb'))

    def __getitem__(self, index):
        raw_data = self._dataset[index]
        output = {**raw_data}
        # print(raw_data)
        '''
        {'event': '1 is trying to wake up on Christmas morning', 
        'img_id': '8PB5sU_QcUc@42', 
        'img_fn': 'movieclips_Carrie_Pilby/8PB5sU_QcUc@42.jpg', 
        'index': 7936, 
        'place': 'in a living room', 
        'task_type': 'before', 
        'labels': "have a poor night's sleep"}
        '''        

        if self._use_image:
            image_dir = os.path.join(self._image_dir, str(raw_data['img_id']) + '.pkl')
            # dict_keys(['image_features', 'object_features'])
            image_data = pickle.load(open(image_dir, 'rb'))

            output['image_features'] = np.row_stack((image_data['image_features'], image_data['object_features']))
            # output['image_features'] = np.row_stack((self.image_data['image_features'], self.image_data['object_features']))


        return output

    def __len__(self):
        return len(self._dataset)

class VCGDataset_gt_retrieval(COCODataset_gt_retrieval):
    def __init__(
            self,
            data_dir,
            image_dir='../vcr/visualcomet/features/',
            split='val',
            generation=None,
            eval_mode=False,
            use_image=True,
            use_event=True,
            pretrain=False,
            index=0,
            num_candidates=50,
            val_data=None,
    ):
        super(VCGDataset_gt_retrieval, self).__init__(
            data_dir=data_dir,
            image_dir=image_dir,
            split=split,
            generation=generation,
            eval_mode=eval_mode,
            use_image=use_image,
            index=index,
            num_candidates=num_candidates,
            val_data=val_data,
        )
        self._use_event = use_event
        self._pretrain = pretrain



    def __getitem__(self, item):
        output = super(VCGDataset_gt_retrieval, self).__getitem__(item)

        person_in_event = re.findall("\d+", output['event']) # max 42
        tmp = []
        for e in output['event'].split():
            if e in person_in_event:
                if int(e) <= 24:
                    e = f'<person{e}> '
            tmp.append(e)
        
        new_event = ' '.join(tmp)
        output['event'] = new_event

        if not self._use_event:
            output['event'] = output['event'].split()[0] # only show the target person
            output['place'] = ' '  
        if self._pretrain:
            output['labels'] = output['event']
            del output['event']
            output['task_type'] = TaskType.CAPTION

        return output


class COCODataset_retrieval(Dataset):
    def __init__(self, data_dir, 
                 image_dir='data/kmbart/', 
                 split='val', 
                 generation=None, 
                 eval_mode=False, 
                 use_image=True, 
                 index=0, 
                 num_candidates=1000,
                 val_data=None,
                 ):
        """
        :param eval_mode: bool, if set to true, "xxx_eval.json" will be loaded, every image will only appear one time
        """
        self._use_image = use_image
        self._data_dir = data_dir
        self._generations = generation
        self._image_dir = data_dir if image_dir is None else image_dir
        self._split = split
        self.index = index

        file_name = split + ('_eval.json' if eval_mode else '.json')
        # file_name = split + ('_eval.json' if eval_mode else '_eval_label.json')
        # self._val = self._generations
        # json.load(open(os.path.join(data_dir, file_name), 'r'))


        # self._val = json.load(open(os.path.join(data_dir, 'val_annots.json'), 'r'))
        val = json.load(open(os.path.join(data_dir, 'val_annots.json'), 'r'))
        self._val = val_data

        '''
        a = json.load(open('/home/user16/HT/KM-BART-ACL/cf_generated/sample_1'))
        a[0]
        {'index': 0, 'task_type': 'intent', 'generations': ['read the note later']}
        '''
        
        dataset = []
        ### image retrieval
        base = deepcopy(self._generations[self.index])
        meta = val[base['index']]
        base['labels'] = base['generations'][0]
        del base['generations']
        base['event'] = meta['event']
        base['place'] = meta['place']
        base['img_id'] = meta['img_fn'].split('/')[-1][:-4]
        base['img_fn'] = meta['img_fn']

        ### text retrieval
        # base = deepcopy(self._val[self.index])

        dataset.append(base)
        # index 0 is always a groud_truth.
        random_idx = random.sample([i for i in range(len(self._val))], num_candidates)
        for i in random_idx:
            data = self._val[i]
            tmp = deepcopy(base)
            # tmp['labels'] = data['labels']
            tmp['event'] = data['event']
            tmp['place'] = data['place']
            tmp['img_id'] = data['img_fn'].split('/')[-1][:-4]
            tmp['img_fn'] = data['img_fn']
            dataset.append(tmp)
            
        self._dataset = dataset

        # self.image_dir = os.path.join(self._image_dir, self._split, str(base['img_id']) + '.pkl')
        # self.image_data = self.image_data = pickle.load(open(self.image_dir, 'rb'))

    def __getitem__(self, index):
        raw_data = self._dataset[index]
        output = {**raw_data}

        if self._use_image:
            image_dir = os.path.join(self._image_dir, self._split, str(raw_data['img_id']) + '.pkl')
            image_data = pickle.load(open(image_dir, 'rb'))
            output['image_features'] = np.concatenate([
                image_data['image_features'],
                image_data['boxes']
            ], axis=1).astype(np.float32)

            if 'mrm_labels' in image_data:
                output['mrm_labels'] = image_data['mrm_labels']

        return output

    def __len__(self):
        return len(self._dataset)


class VCGDataset_retrieval(COCODataset_retrieval):
    def __init__(
            self,
            data_dir,
            image_dir='data/kmbart/',
            split='val',
            generation=None,
            eval_mode=False,
            use_image=True,
            use_event=True,
            pretrain=False,
            index=0,
            num_candidates=50,
            val_data=None,
    ):
        super(VCGDataset_retrieval, self).__init__(
            data_dir=data_dir,
            image_dir=image_dir,
            split=split,
            generation=generation,
            eval_mode=eval_mode,
            use_image=use_image,
            index=index,
            num_candidates=num_candidates,
            val_data=val_data,
        )
        self._use_event = use_event
        self._pretrain = pretrain

    def __getitem__(self, item):
        output = super(VCGDataset_retrieval, self).__getitem__(item)

        # ## 2022-09-25까지 사용
        # person_in_event = re.findall("\d+", output['event']) # max 42
        # tmp = []
        # for e in output['event'].split():
        #     if e in person_in_event:
        #         if int(e) <= 42:
        #             e = 'person ' + e
        #     tmp.append(e)


        # output['event'] = ' '.join(tmp)
        
        if not self._use_event:
            output['event'] = output['event'].split()[0] # only show the target person
            output['place'] = ' '  
        if self._pretrain:
            output['labels'] = output['event']
            del output['event']
            output['task_type'] = TaskType.CAPTION

        return output
