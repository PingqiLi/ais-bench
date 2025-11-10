# Copyright (c) OpenDataLab https://github.com/opendatalab/OmniDocBench (2025/11/07)
# SPDX-License-Identifier: Apache-2.0
# Part of this document is directly reused from the above warehouse without modification.

import re
from datetime import datetime
import os 
import json
import copy
import time
import logging
import Levenshtein
from tqdm import tqdm
import evaluate
import numpy as np
import random
import shutil
from collections import deque, defaultdict
from apted.helpers import Tree
from apted import APTED, Config
from lxml import etree, html
import pandas as pd
import subprocess
from skimage.measure import ransac
from threading import Timer
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, as_completed

from ais_bench.benchmark.datasets.omnidocbench.registory import METRIC_REGISTRY


SKIP_PATTERNS = [r'\{', r'\}', r'[\[\]]', r'\\begin\{.*?\}', r'\\end\{.*?\}', r'\^', r'\_', r'\\.*rule.*', r'\\.*line.*', r'\[[\-.0-9]+[epm][xtm]\]']
SKIP_Tokens = ['\\', '\\\\', '\\index', '\\a', '&', '$', '\\multirow', '\\def', '\\raggedright', '\\url', '\\cr', '\\ensuremath', '\\left', '\\right', 
               '\\mathchoice', '\\scriptstyle', '\\displaystyle', '\\qquad', '\\quad', '\\,', '\\!', '~', '\\boldmath']
PHANTOM_Tokens = ['\\fontfamily', '\\vphantom', '\\phantom', '\\rowcolor', '\\ref']
TWO_Tail_Tokens = ['\\frac', '\\binom']
AB_Tail_Tokens = ['\\xrightarrow', '\\xleftarrow', '\\sqrt']        # special token \xxx [] {} 
TWO_Tail_Invisb_Tokens = ['\\overset', '\\underset', '\\stackrel']
ONE_Tail_Tokens = ['\\widetilde', '\\overline', '\\hat', '\\widehat', '\\tilde', '\\Tilde', '\\dot', '\\bar', '\\vec', '\\underline', '\\underbrace', '\\check',
                   '\\breve', '\\Bar', '\\Vec', '\\mathring', '\\ddot']
ONE_Tail_Invisb_Tokens = ['\\boldsymbol', '\\pmb', '\\textbf', '\\mathrm', '\\mathbf', '\\mathbb', '\\mathcal', '\\textmd', '\\texttt', '\\textnormal', 
                          '\\text', '\\textit', '\\textup', '\\mathop', '\\mathbin', '\\smash', '\\operatorname', '\\textrm', '\\mathfrak', '\\emph',
                          '\\textsf', '\\textsc']

tabular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amssymb}
\usepackage{upgreek}
\usepackage{amsmath}
\usepackage{xcolor}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

# Need to configure Source Han Sans SC or other Chinese fonts
formular_template = r"""
\documentclass[12pt]{article}
\usepackage[landscape]{geometry}
\usepackage{geometry}
\geometry{a<PaperSize>paper,scale=0.98}
\pagestyle{empty}
\usepackage{amsmath}
\usepackage{upgreek}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{xeCJK}
\setCJKmainfont{Source Han Sans SC}
\setCJKsansfont{Source Han Sans SC}
\setCJKmonofont{Source Han Sans SC}
\xeCJKsetup{CJKmath=true}
\begin{document}
\makeatletter
\renewcommand*{\@textcolor}[3]{%%
  \protect\leavevmode
  \begingroup
    \color#1{#2}#3%%
  \endgroup
}
\makeatother
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            if self.colspan is None or self.rowspan is None:
                result = '"tag": %s, "colspan": None, "rowspan": None, "text": %s' % \
                     (self.tag, self.content)
            else:
                result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                        (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in list(node):
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        scores = dict(zip(samples, scores))
        return scores

@METRIC_REGISTRY.register("TEDS")
class call_TEDS():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        teds = TEDS(structure_only=False)
        teds_structure_only = TEDS(structure_only=True)
        group_scores = defaultdict(list)
        group_scores_structure_only = defaultdict(list)
        samples = self.samples
        per_table_score = {}
        for sample in samples:
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            try:
                score = teds.evaluate(pred, gt)
            except:
                score = 0
                print(f'TEDS score error for table {sample["gt_idx"]} in {sample["img_id"]}. The score is set to 0.')
            try:
                score_structure_only = teds_structure_only.evaluate(pred, gt)
            except:
                score_structure_only = 0
                print(f'TEDS_structure_only score error for table {sample["gt_idx"]} in {sample["img_id"]}. The score is set to 0.')
            group_scores['all'].append(score)
            group_scores_structure_only['all'].append(score_structure_only)
            if not sample.get('metric'):
                sample['metric'] = {}
            sample['metric']['TEDS'] = score
            sample['metric']['TEDS_structure_only'] = score_structure_only
            per_table_score[sample['img_id']+'_'+str(sample['gt_idx'])] = {'TEDS': score, 'TEDS_structure_only': score_structure_only}
            for group in group_info:
                select_flag = True
                for k, v in group.items():
                    for gt_attribute in sample['gt_attribute']:   # gt_attribute is a list containing all merged gt attributes
                        if not gt_attribute:   # if no GT attributes, don't include in calculation
                            select_flag = False
                        elif gt_attribute[k] != v:  # if any gt attribute doesn't meet criteria, don't select
                            select_flag = False
                if select_flag:
                    group_scores[str(group)].append(score)
        with open(f'./result/{save_name}_per_table_TEDS.json', 'w', encoding='utf-8') as f:
            json.dump(per_table_score, f, indent=4, ensure_ascii=False)
        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                result[group_name] = 'NaN'
                print(f'Warning: Empyty matched samples for {group_name}.')
        
        structure_only_result = {}
        for group_name, scores in group_scores_structure_only.items():
            if len(scores) > 0:
                structure_only_result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                structure_only_result[group_name] = 'NaN'
                print(f'Warning: Empyty matched samples for {group_name}.')

        return samples, {'TEDS': result, 'TEDS_structure_only': structure_only_result}


def get_groups(samples, group_info):
    group_samples = defaultdict(list)
    for sample in samples:
        group_samples['all'].append(sample)
        for group in group_info:
            select_flag = True
            for k, v in group.items():
                for gt_attribute in sample['gt_attribute']:   # gt_attribute is a list containing all merged gt attributes
                    if not gt_attribute:   # if no GT attributes, don't include in calculation
                        select_flag = False
                    elif gt_attribute[k] != v:  # if any gt attribute doesn't meet criteria, don't select
                        select_flag = False
            if select_flag:
                group_samples[str(group)].append(sample)
    return group_samples


@METRIC_REGISTRY.register("BLEU")
class call_BLEU():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        group_samples = get_groups(self.samples, group_info)
        result = {}
        for group_name, samples in group_samples.items():
            predictions, references = [], []
            for sample in samples:
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                predictions.append(gt)
                references.append(pred)
            bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1,1e8))
            bleu_results = bleu.compute(predictions=predictions, references=references)
            result[group_name] = bleu_results["bleu"]
        
        return self.samples, {'BLEU': result}
    
@METRIC_REGISTRY.register("METEOR")
class call_METEOR():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        group_samples = get_groups(self.samples, group_info)
        result = {}
        for group_name, samples in group_samples.items():
            predictions, references = [], []
            for sample in samples:
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                predictions.append(gt)
                references.append(pred)
            meteor = evaluate.load('meteor', keep_in_memory=True, experiment_id=random.randint(1,1e8))
            meteor_results = meteor.compute(predictions=predictions, references=references)
            result[group_name] = meteor_results['meteor']
        
        return self.samples, {'METEOR': result}

@METRIC_REGISTRY.register("Edit_dist")
class call_Edit_dist():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', out_dir='./'):
        samples = self.samples
        for sample in samples:
            img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith('.png') else '_'.join(sample['img_id'].split('_')[:-1])
            sample['image_name'] = img_name
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            upper_len = max(len(pred), len(gt))
            sample['upper_len'] = upper_len
            if len(pred) > 0 or len(gt) > 0:
                edit_dist = Levenshtein.distance(pred, gt)
                if not sample.get('metric'):
                    sample['metric'] = {}
                sample['metric']['Edit_dist'] = edit_dist / upper_len
                sample['Edit_num'] = edit_dist

        if isinstance(samples, list):
            saved_samples = samples
        else:
            saved_samples = samples.samples
        
        if not saved_samples:
            return samples, {'Edit_dist': {'ALL_page_avg': 'NaN'}}

        df = pd.DataFrame(saved_samples)
        up_total_avg = df.groupby("image_name").apply(lambda x: x['Edit_num'].sum() / x['upper_len'].sum()) # page level, sum of edits divided by sum of max(gt,pred) lengths for each sample
        all_total_avg = df['Edit_num'].sum() / df['upper_len'].sum()
        per_img_score = up_total_avg.to_dict()
        with open(f'{out_dir}/{save_name}_per_page_edit.json', 'w', encoding='utf-8') as f:  #xieruwenjian 
            json.dump(per_img_score, f, indent=4, ensure_ascii=False)        
        
        edit_whole = df['Edit_num'].sum() / df['upper_len'].sum()
        df['ratio'] = df['Edit_num'] / df['upper_len']
        edit_sample_avg = df['ratio'].mean()
        # edit_sample_avg = df['metric']['Edit_dist'].mean()
        return samples, {'Edit_dist': {'ALL_page_avg': up_total_avg.mean(), 'edit_whole': edit_whole, 'edit_sample_avg': edit_sample_avg}}


def norm_same_token(token):
    special_map = {
        "\\cdot": ".",
        "\\mid": "|",
        "\\to": "\\rightarrow",
        "\\top": "T",
        "\\Tilde": "\\tilde",
        "\\cdots": "\\dots",
        "\\prime": "'",
        "\\ast": "*",
        "\\left<": "\\langle",
        "\\right>": "\\rangle"
    }
    if token in special_map.keys():
        token = special_map[token]
    if token.startswith('\\left') or token.startswith('\\right'):
        token = token.replace("\\left", "").replace("\\right", "")
    if token.startswith('\\big') or token.startswith('\\Big'):
        if "\\" in token[4:]:
            token = "\\"+token[4:].split("\\")[-1]
        else:
            token = token[-1]
    
    if token in ['\\leq', '\\geq']:
        return token[0:-1]
    if token in ['\\lVert', '\\rVert', '\\Vert']:
        return '\\|'
    if token in ['\\lvert', '\\rvert', '\\vert']:
        return '|'
    if token.endswith("rightarrow"):
        return "\\rightarrow"
    if token.endswith("leftarrow"):
        return "\\leftarrow"
    if token.startswith('\\wide'):
        return token.replace("wide", "")
    if token.startswith('\\var'):
        return token.replace("\\var", "")
    return token


class HungarianMatcher:
    def __init__(
        self, 
        cost_token: float = 1,
        cost_position: float = 0.05,
        cost_order: float = 0.15,
    ):
        self.cost_token = cost_token
        self.cost_position = cost_position
        self.cost_order = cost_order
        self.cost = {}
    
    def calculate_token_cost_old(self, box_gt, box_pred):
        token_cost = np.ones((len(box_gt), len(box_pred)))
        for i in range(token_cost.shape[0]):
            box1 = box_gt[i]
            for j in range(token_cost.shape[1]):
                box2 = box_pred[j]
                if box1['token'] == box2['token']:
                    token_cost[i, j] = 0
                elif norm_same_token(box1['token']) == norm_same_token(box2['token']):
                    token_cost[i, j] = 0.05
        return np.array(token_cost)
        
    def calculate_token_cost(self, box_gt, box_pred):
        token2id = {}
        for data in box_gt+box_pred:
            if data['token'] not in token2id:
                token2id[data['token']] = len(token2id)
        num_classes = len(token2id)
        
        token2id_norm = {}
        for data in box_gt+box_pred:
            if norm_same_token(data['token']) not in token2id_norm:
                token2id_norm[norm_same_token(data['token'])] = len(token2id_norm)
        num_classes_norm = len(token2id_norm)
        
        gt_token_array = []
        norm_gt_token_array = []    
        for data in box_gt:
            gt_token_array.append(token2id[data['token']])
            norm_gt_token_array.append(token2id_norm[norm_same_token(data['token'])])
            
        pred_token_logits = []
        norm_pred_token_logits = []
        for data in box_pred:
            logits = [0] * num_classes
            logits[token2id[data['token']]] = 1
            pred_token_logits.append(logits)
            
            logits_norm = [0] * num_classes_norm
            logits_norm[token2id_norm[norm_same_token(data['token'])]] = 1
            norm_pred_token_logits.append(logits_norm)
            
        gt_token_array = np.array(gt_token_array)
        pred_token_logits = np.array(pred_token_logits)
        
        norm_gt_token_array = np.array(norm_gt_token_array)
        norm_pred_token_logits = np.array(norm_pred_token_logits)
        
        token_cost = 1.0 - pred_token_logits[:, gt_token_array]
        norm_token_cost = 1.0 - norm_pred_token_logits[:, norm_gt_token_array]

        token_cost[np.logical_and(token_cost==1, norm_token_cost==0)] = 0.05
        return token_cost.T
        
        
    def box2array(self, box_list, size):
        W, H = size
        box_array = []
        for box in box_list:
            x_min, y_min, x_max, y_max = box['bbox']
            box_array.append([x_min/W, y_min/H, x_max/W, y_max/H])
        return np.array(box_array)
        
    def order2array(self, box_list):
        order_array = []
        for idx, box in enumerate(box_list):
            order_array.append([idx / len(box_list)])
        return np.array(order_array)
    
    def calculate_l1_cost(self, gt_array, pred_array):
        scale = gt_array.shape[-1]
        l1_cost = cdist(gt_array, pred_array, 'minkowski', p=1)
        return l1_cost / scale
        
    def __call__(self, box_gt, box_pred, gt_size, pred_size):
        aa = time.time()
        gt_box_array = self.box2array(box_gt, gt_size)
        pred_box_array = self.box2array(box_pred, pred_size)
        gt_order_array = self.order2array(box_gt)
        pred_order_array = self.order2array(box_pred)

        token_cost = self.calculate_token_cost(box_gt, box_pred)
        position_cost = self.calculate_l1_cost(gt_box_array, pred_box_array)
        order_cost = self.calculate_l1_cost(gt_order_array, pred_order_array)

        self.cost["token"] = token_cost
        self.cost["position"] = position_cost
        self.cost["order"] = order_cost
        
        cost = self.cost_token * token_cost + self.cost_position * position_cost + self.cost_order * order_cost
        cost[np.isnan(cost) | np.isinf(cost)] = 100
        indexes = linear_sum_assignment(cost)
        matched_idxes = []
        for a, b in zip(*indexes):
            matched_idxes.append((a, b))
        
        return matched_idxes


def run_cmd(cmd, timeout_sec=30, temp_dir=None):
    # 设置进程独立的环境变量
    env = os.environ.copy()
    if temp_dir:
        env['TMPDIR'] = temp_dir
        env['TMP'] = temp_dir  
        env['TEMP'] = temp_dir
        env['MAGICK_TMPDIR'] = temp_dir
        env['TEXMFCACHE'] = temp_dir
        env['TEXMFVAR'] = temp_dir
    try:
        proc = subprocess.Popen(cmd, shell=True, env=env)
    except Exception as e:
        logging.warning(e)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout,stderr = proc.communicate()
    finally:
        timer.cancel()
        
def convert_pdf2img(pdf_filename, png_filename, temp_dir=None):
    cmd = "magick -density 200 -quality 100 \"%s\" \"%s\""%(pdf_filename, png_filename)
    run_cmd(cmd, temp_dir=temp_dir)

def crop_image(image_path, pad=8):
    img = Image.open(image_path).convert("L")
    img_data = np.asarray(img, dtype=np.uint8)
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        y_min = 0
        y_max = 10
        x_min = 0
        x_max = 10
    else:
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])
        
    img = Image.open(image_path).convert("RGB").crop((x_min-pad, y_min-pad, x_max+pad, y_max+pad))
    img.save(image_path)
    
def extrac_bbox_from_color_image(image_path, color_list):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    pixels = list(img.getdata())
    
    bbox_list = []
    for target_color in color_list:
        target_pixels = [ i for i, pixel in enumerate(pixels)if pixel == target_color ]
        x_list = []
        y_list = []
        for idx in target_pixels:
            x_list.append(idx % W)
            y_list.append(idx // W)
        try:
            y_min, y_max, x_min, x_max = min(y_list), max(y_list), min(x_list), max(x_list)
            bbox_list.append([x_min-1, y_min-1, x_max+1, y_max+1])

        except:
            bbox_list.append([])
            continue
        
    img = img.convert("L")
    img_bw = img.point(lambda x: 255 if x == 255 else 0, '1')
    img_bw.convert("RGB").save(image_path) 
    return bbox_list