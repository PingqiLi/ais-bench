# Copyright (c) OpenDataLab https://github.com/opendatalab/OmniDocBench (2025/11/07)
# SPDX-License-Identifier: Apache-2.0
# Part of this document is directly reused from the above warehouse without modification.

import re
import os
import sys
import shutil
from copy import deepcopy
import copy
from typing import List, Dict, Any
from collections import defaultdict, Counter
import html
import uuid
import unicodedata
import subprocess
import pandas as pd
from scipy.optimize import linear_sum_assignment
import Levenshtein
import numpy as np
from bs4 import BeautifulSoup
from Levenshtein import distance as Levenshtein_distance


inline_reg = re.compile(
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',
)
ARRAY_RE = re.compile(
    r'\\begin\{array\}\{(?P<spec>[^}]*)\}(?P<body>.*?)\\end\{array\}',
    re.S
)

# img
img_pattern = r'!\[.*?\]\(.*?\)'

# table 
table_reg = re.compile(
    r'\\begin{table\*?}(.*?)\\end{table\*?}|'
    r'\\begin{tabular\*?}(.*?)\\end{tabular\*?}',
    re.DOTALL 
)
md_table_reg = re.compile(
    r'\|\s*.*?\s*\|\n', 
    re.DOTALL)
html_table_reg = re.compile(
    r'(<table.*?</table>)',
    re.DOTALL
)

# title
title_reg = re.compile(
    r'^\s*#.*$', 
    re.MULTILINE)

# code block
code_block_reg = re.compile(
    r'```(\w+)\n(.*?)```',
    re.DOTALL
)

display_reg = re.compile(
    # r'\\begin{equation\*?}(.*?)\\end{equation\*?}|'
    # r'\\begin{align\*?}(.*?)\\end{align\*?}|'
    # r'\\begin{gather\*?}(.*?)\\end{gather\*?}|'
    # r'\\begin{array\*?}(.*?)\\end{array\*?}|'
    r'\$\$(.*?)\$\$|'
    r'\\\[(.*?)\\\]|'
    r'\$(.*?)\$|'
    r'\\\((.*?)\\\)',  
    re.DOTALL
)


def remove_markdown_fences(content):
    content = re.sub(r'^```markdown\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```html\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'^```latex\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'```\n?$', '', content, flags=re.MULTILINE)
    return content

# Standardize all consecutive characters
def replace_repeated_chars(input_str):
    input_str = re.sub(r'_{4,}', '____', input_str) # Replace more than 4 consecutive underscores with 4 underscores
    input_str = re.sub(r' {4,}', '    ', input_str)   # Replace more than 4 consecutive spaces with 4 spaces
    return input_str
    # return re.sub(r'([^a-zA-Z0-9])\1{10,}', r'\1\1\1\1', input_str) # For other consecutive symbols (except numbers and letters), replace more than 10 occurrences with 4

def extract_tabular(text):
    begin_pattern = r'\\begin{tabular}'
    end_pattern = r'\\end{tabular}'

    tabulars = []
    positions = []
    current_pos = 0
    stack = []
    
    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])
        
        if not begin_match and not end_match:
            break
            
        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1
    
    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_tabular(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions

def extract_tex_table(content):
    tables = []
    tables_positions = []

    pattern = r'\\begin{table}(.*?)\\end{table}'
    for match in re.finditer(pattern, content, re.DOTALL):
        start_pos = match.start()
        end_pos = match.end()
        table_content = match.group(0)
        tables.append(table_content)
        tables_positions.append((start_pos, end_pos))
        content = content[:start_pos] + ' '*(end_pos-start_pos) + content[end_pos:]

    tabulars, tabular_positions = extract_tabular(content)
    all_tables = tables + tabulars
    all_positions = tables_positions + tabular_positions

    all_result = sorted([[pos, table]for pos, table in zip(all_positions, all_tables)], key=lambda x: x[0][0])
    all_tables = [x[1] for x in all_result]
    all_positions = [x[0] for x in all_result]

    return all_tables, all_positions

def extract_html_table(text):
    begin_pattern = r'<table(?:[^>]*)>'
    end_pattern = r'</table>'

    tabulars = []
    positions = []
    current_pos = 0
    stack = []
    
    while current_pos < len(text):
        begin_match = re.search(begin_pattern, text[current_pos:])
        end_match = re.search(end_pattern, text[current_pos:])
        
        if not begin_match and not end_match:
            break
            
        if begin_match and (not end_match or begin_match.start() < end_match.start()):
            stack.append(current_pos + begin_match.start())
            current_pos += begin_match.start() + len(end_pattern)
        elif end_match:
            if stack:
                start_pos = stack.pop()
                if not stack:
                    end_pos = current_pos + end_match.start() + len(end_pattern)
                    tabular_code = text[start_pos:end_pos]
                    tabulars.append(tabular_code)
                    positions.append((start_pos, end_pos))
            current_pos += end_match.start() + len(end_pattern)
        else:
            current_pos += 1
    
    if stack:
        new_start = stack[0] + len(begin_pattern)
        new_tabulars, new_positions = extract_html_table(text[new_start:])
        new_positions = [(start + new_start, end + new_start) for start, end in new_positions]
        tabulars.extend(new_tabulars)
        positions.extend(new_positions)

    return tabulars, positions

# The function is from https://github.com/intsig-textin/markdown_tester
def markdown_to_html(markdown_table):
    rows = [row.strip() for row in markdown_table.strip().split('\n')]
    num_columns = len(rows[0].split('|')) - 2

    html_table = '<table>\n  <thead>\n    <tr>\n'

    header_cells = [cell.strip() for cell in rows[0].split('|')[1:-1]]
    for cell in header_cells:
        html_table += f'      <th>{cell}</th>\n'
    html_table += '    </tr>\n  </thead>\n  <tbody>\n'

    for row in rows[2:]:
        cells = [cell.strip() for cell in row.split('|')[1:-1]]
        html_table += '    <tr>\n'
        for cell in cells:
            html_table += f'      <td>{cell}</td>\n'
        html_table += '    </tr>\n'

    html_table += '  </tbody>\n</table>\n'
    return html_table

def convert_table(input_str):
    # Replace <table>
    output_str = input_str.replace("<table>", "<table border=\"1\" >")

    # Replace <td>
    output_str = output_str.replace("<td>", "<td colspan=\"1\" rowspan=\"1\">")

    return output_str

def find_md_table_mode(line):
    if re.search(r'-*?:',line) or re.search(r'---',line) or re.search(r':-*?',line):
        return True
    return False

def delete_table_and_body(input_list):
    res = []
    for line in input_list:
        if not re.search(r'</?t(able|head|body)>',line):
            res.append(line)
    return res

def merge_tables(input_str):
    # Delete HTML comments
    input_str = re.sub(r'<!--[\s\S]*?-->', '', input_str)

    # Use regex to find each <table> block
    table_blocks = re.findall(r'<table>[\s\S]*?</table>', input_str)

    # Process each <table> block, replace <th> with <td>
    output_lines = []
    for block in table_blocks:
        block_lines = block.split('\n')
        for i, line in enumerate(block_lines):
            if '<th>' in line:
                block_lines[i] = line.replace('<th>', '<td>').replace('</th>', '</td>')
        final_tr = delete_table_and_body(block_lines)
        if len(final_tr) > 2:
            output_lines.extend(final_tr)  # Ignore <table> and </table> tags, keep only table content

    # Rejoin the processed strings
    merged_output = '<table>\n{}\n</table>'.format('\n'.join(output_lines))

    return "\n\n" + merged_output + "\n\n"

def replace_table_with_placeholder(input_string):
    lines = input_string.split('\n')
    output_lines = []

    in_table_block = False
    temp_block = ""
    last_line = ""

    for idx, line in enumerate(lines):
        if "<table>" in line:
            in_table_block = True
            temp_block += last_line
        elif in_table_block:
            if not find_md_table_mode(last_line) and "</thead>" not in last_line:
                temp_block += "\n" + last_line
            if "</table>" in last_line:
                if "<table>" not in line:
                    in_table_block = False
                    output_lines.append(merge_tables(temp_block))
                    temp_block = ""
        else:
            output_lines.append(last_line)

        last_line = line
    if last_line:
        if in_table_block or "</table>" in last_line:
            temp_block += "\n" + last_line
            output_lines.append(merge_tables(temp_block))
        else:
            output_lines.append(last_line)

    return '\n'.join(output_lines)


def convert_markdown_to_html(markdown_content):
    # Define a regex pattern to find Markdown tables with newlines
    markdown_content = markdown_content.replace('\r', '')+'\n'
    pattern = re.compile(r'\|\s*.*?\s*\|\n', re.DOTALL)

    # Find all matches in the Markdown content
    matches = pattern.findall(markdown_content)

    for match in matches:
        html_table = markdown_to_html(match)
        markdown_content = markdown_content.replace(match, html_table, 1)  # Only replace the first occurrence

    res_html = convert_table(replace_table_with_placeholder(markdown_content))

    return res_html

def md_tex_filter(content):
    '''
    Input: 1 page md or tex content - String
    Output: text, display, inline, table, title, code - list
    '''
    content = re.sub(img_pattern, '', content)  # remove image
    content = remove_markdown_fences(content)   # remove markdown fences
    content = replace_repeated_chars(content) # replace all consecutive characters
    content = content.replace('<html>', '').replace('</html>', '').replace('<body>', '').replace('</body>', '')
    pred_all = []
    latex_table_array, table_positions = extract_tex_table(content)
    for latex_table, position in zip(latex_table_array, table_positions):
        position = [position[0], position[0]+len(latex_table)]
        pred_all.append({
            'category_type': 'latex_table',
            'position': position,
            'content': latex_table
        })
        content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace latex table with space

    # extract html table  
    html_table_array, table_positions = extract_html_table(content)
    for html_table, position in zip(html_table_array, table_positions):
        position = [position[0], position[0]+len(html_table)]
        pred_all.append({
            'category_type': 'html_table',
            'position': position,
            'content': html_table
        })
        content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace html table with space

    # extract interline formula
    display_matches = display_reg.finditer(content)
    content_copy = content
    for match in display_matches:
        matched = match.group(0)
        if matched:
            # single_line = ''.join(matched.split())
            single_line = ' '.join(matched.strip().split('\n'))
            position = [match.start(), match.end()]
            # replace $$ with \[\]
            dollar_pattern = re.compile(r'\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)', re.DOTALL)
            sub_match = dollar_pattern.search(single_line)
            if sub_match is None:
                # pass
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line
                })
            elif sub_match.group(1):
                single_line = re.sub(dollar_pattern, r'\\[\1\\]', single_line)
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace equation with space
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line
                })
            else:
                single_line = re.sub(dollar_pattern, r'\\[\2\3\\]', single_line)
                pred_all.append({
                    'category_type': 'equation_isolated',
                    'position': position,
                    'content': single_line,
                    'fine_category_type': 'equation_inline'
                })

    md_table_mathces = md_table_reg.findall(content+'\n')
    if len(md_table_mathces) >= 2:
        content = convert_markdown_to_html(content)
        html_table_matches = html_table_reg.finditer(content)
        if html_table_matches:
            for match in html_table_matches:
                matched = match.group(0)
                position = [match.start(), match.end()]
                content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace md table with space
                pred_all.append({
                    'category_type': 'html_table',
                    'position': position,
                    'content': matched.strip(),
                    'fine_category_type': 'md2html_table'
                })

    # extract code blocks
    code_matches = code_block_reg.finditer(content)
    if code_matches:
        for match in code_matches:
            position = [match.start(), match.end()]
            language = match.group(1)
            code = match.group(2).strip()
            content = content[:position[0]] + ' '*(position[1]-position[0]) + content[position[1]:]  # replace code block with space
            pred_all.append({
                'category_type': 'text_all',
                'position': position,
                'content': code,
                'language': language,
                'fine_category_type': 'code'
            })

    # Remove latex style
    content = re.sub(r'\\title\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\title\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\\section\*?\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\section\*?\{\s*(.*?)\s*\}', r'\1', content, flags=re.DOTALL)

    # extract texts
    res = content.split('\n\n')
    if len(res) == 1:
        res = content.split('\n')  # some models do not use double newlines, so use single newlines to split

    content_position = 0
    for text in res:
        position = [content_position, content_position+len(text)]
        content_position += len(text)
        text = text.strip()
        text = text.strip('\n')
        text = '\n'.join([_.strip() for _ in text.split('\n') if _.strip()])   # avoid some single newline content with many spaces

        if text:  # Check if the stripped text is not empty
            if text.startswith('<table') and text.endswith('</table>'):
                pred_all.append({
                    'category_type': 'html_table',
                    'position': position,
                    'content': text,
                })
            elif text.startswith('$') and text.endswith('$'):
                if text.replace('$', '').strip():
                    pred_all.append({
                        'category_type': 'equation_isolated',
                        'position': position,
                        'content': text.strip(),
                    })
            else:
                text = text.strip()
                if text:
                    pred_all.append({
                        'category_type': 'text_all',
                        'position': position,
                        'content': text,
                        'fine_category_type': 'text_block'
                    })

    pred_dataset = defaultdict(list)
    pred_all = sorted(pred_all, key=lambda x: x['position'][0])
    for item in pred_all:
        pred_dataset[item['category_type']].append(item)
    return pred_dataset

def normalized_formula(text):
    # Normalize math formulas before matching
    filter_list = ['\\mathbf', '\\mathrm', '\\mathnormal', '\\mathit', '\\mathbb', '\\mathcal', '\\mathscr', '\\mathfrak', '\\mathsf', '\\mathtt', 
                   '\\textbf', '\\text', '\\boldmath', '\\boldsymbol', '\\operatorname', '\\bm',
                   '\\symbfit', '\\mathbfcal', '\\symbf', '\\scriptscriptstyle', '\\notag',
                   '\\setlength', '\\coloneqq', '\\space', '\\thickspace', '\\thinspace', '\\medspace', '\\nobreakspace', '\\negmedspace',
                   '\\quad', '\\qquad', '\\enspace', '\\substackw', ' ', '$$', '\\left', '\\right', '\\displaystyle', '\\text']
    
    # delimiter_filter
    text = text.strip().strip('$').strip('\n')
    pattern = re.compile(r"\\\[(.+?)(?<!\\)\\\]")
    match = pattern.search(text)

    if match:
        text = match.group(1).strip()
    
    tag_pattern = re.compile(r"\\tag\{.*?\}")
    text = tag_pattern.sub('', text)
    hspace_pattern = re.compile(r"\\hspace\{.*?\}")
    text = hspace_pattern.sub('', text)
    begin_pattern = re.compile(r"\\begin\{.*?\}")
    text = begin_pattern.sub('', text)
    end_pattern = re.compile(r"\\end\{.*?\}")
    text = end_pattern.sub('', text)
    col_sep = re.compile(r"\\arraycolsep.*?\}")
    text = col_sep.sub('', text)
    text = text.strip('.')
    
    for filter_text in filter_list:
        text = text.replace(filter_text, '')
    text = text.lower()
    return text

def textblock2unicode(text):
    from pylatexenc.latex2text import LatexNodes2Text
    inline_matches = inline_reg.finditer(text)
    removal_positions = []
    for match in inline_matches:
        position = [match.start(), match.end()]
        content = match.group(1) if match.group(1) is not None else match.group(2)
        # Remove escape characters \
        clean_content = re.sub(r'\\([\\_&%^])', '', content)

        try:
            if any(char in clean_content for char in r'\^_'):
                if clean_content.endswith('\\'):
                    clean_content += ' '
                # inline_array.append(match.group(0))
                unicode_content = LatexNodes2Text().latex_to_text(clean_content)
                removal_positions.append((position[0], position[1], unicode_content))
        except:
            continue
    
    # Remove inline formulas from original text
    for start, end, unicode_content in sorted(removal_positions, reverse=True):
        text = text[:start] + unicode_content.strip() + text[end:]

    return text

# Text OCR quality check processing:
def clean_string(input_string):
    # Use regex to keep Chinese characters, English letters and numbers
    # input_string = input_string.replace('\\t', '').replace('\\n', '').replace('\t', '').replace('\n', '').replace('/t', '').replace('/n', '')
    input_string = input_string.replace('\\t', '').replace('\\n', '').replace('\t', '').replace('\n', '').replace('/t', '').replace('/n', '')
    cleaned_string = re.sub(r'[^\w\u4e00-\u9fff]', '', input_string)   # 只保留中英文和数字
    return cleaned_string

def get_pred_category_type(pred_idx, pred_items):
    try:
        if pred_items[pred_idx].get('fine_category_type'):
            pred_pred_category_type = pred_items[pred_idx]['fine_category_type']
        else:
            pred_pred_category_type = pred_items[pred_idx]['category_type']
        return pred_pred_category_type
    except:
        return ""

def compute_edit_distance_matrix_new(gt_lines, matched_lines):
    try:
        distance_matrix = np.zeros((len(gt_lines), len(matched_lines)))
        for i, gt_line in enumerate(gt_lines):
            for j, matched_line in enumerate(matched_lines):
                if len(gt_line) == 0 and len(matched_line) == 0:
                    distance_matrix[i][j] = 0  
                else:
                    distance_matrix[i][j] = Levenshtein.distance(gt_line, matched_line) / max(len(matched_line), len(gt_line))
        return distance_matrix
    except ZeroDivisionError:
        raise  


## mix match
def get_gt_pred_lines(gt_mix,pred_dataset_mix,line_type):

    norm_html_lines,gt_lines,pred_lines,norm_gt_lines,norm_pred_lines,gt_cat_list = [],[],[],[],[],[]
    if line_type in ['html_table','latex_table']:
        for item in gt_mix:
            if item.get('fine_category_type'):
                gt_cat_list.append(item['fine_category_type'])
            else:
                gt_cat_list.append(item['category_type'])
            if item.get('content'):
                gt_lines.append(str(item['content']))
                norm_html_lines.append(str(item['content']))
            elif line_type == 'text':
                gt_lines.append(str(item['text']))
            elif line_type == 'html_table':
                gt_lines.append(str(item['html']))
            elif line_type == 'formula':
                gt_lines.append(str(item['latex']))
            elif line_type == 'latex_table':
                gt_lines.append(str(item['latex']))
                norm_html_lines.append(str(item['html']))
        
        pred_lines = [str(item['content']) for item in pred_dataset_mix]
        if line_type == 'formula':
            norm_gt_lines = [normalized_formula(_) for _ in gt_lines]
            norm_pred_lines = [normalized_formula(_) for _ in pred_lines]
        elif line_type == 'text':
            norm_gt_lines = [clean_string(textblock2unicode(_)) for _ in gt_lines]
            norm_pred_lines = [clean_string(textblock2unicode(_)) for _ in pred_lines]
        else:
            norm_gt_lines = gt_lines
            norm_pred_lines = pred_lines
        if line_type == 'latex_table':
            gt_lines = norm_html_lines

    else:
        for item in pred_dataset_mix:
            # text
            if item['category_type'] == 'text_all':
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(clean_string(textblock2unicode(str(item['content']))))
            # formula
            elif  item['category_type']=='equation_isolated':
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(normalized_formula(str(item['content'])))
            # table
            else:
                pred_lines.append(str(item['content']))
                norm_pred_lines.append(str(item['content']))
        
        for item in gt_mix:
            if item.get('content'):
                gt_lines.append(str(item['content']))
                if item['category_type'] == 'text_all':              
                    norm_gt_lines.append(clean_string(textblock2unicode(str(item['content']))))
                else:
                   norm_gt_lines.append(item['content'])
                
                norm_html_lines.append(str(item['content']))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])
            # text      
            elif item['category_type'] in ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption','figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption','header', 'footer', 'page_footnote', 'page_number']:
                gt_lines.append(str(item['text']))
                norm_gt_lines.append(clean_string(textblock2unicode(str(item['text']))))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])

            # formula
            elif item['category_type'] == 'equation_isolated':
                gt_lines.append(str(item['latex']))
                norm_gt_lines.append(normalized_formula(str(item['latex'])))

                if item.get('fine_category_type'):
                    gt_cat_list.append(item['fine_category_type'])
                else:
                    gt_cat_list.append(item['category_type'])


    filtered_lists = [(a, b, c) for a, b, c in zip(gt_lines, norm_gt_lines, gt_cat_list) if a and b]

    # decompress to three lists
    if filtered_lists:
        gt_lines_c, norm_gt_lines_c, gt_cat_list_c = zip(*filtered_lists)

        # convert to lists
        gt_lines_c = list(gt_lines_c)
        norm_gt_lines_c = list(norm_gt_lines_c)
        gt_cat_list_c = list(gt_cat_list_c)
    else:
        gt_lines_c = []
        norm_gt_lines_c = []
        gt_cat_list_c = []

    # pred's empty values
    filtered_lists = [(a, b) for a, b in zip(pred_lines, norm_pred_lines) if a and b]

    # decompress to two lists
    if filtered_lists:
        pred_lines_c, norm_pred_lines_c = zip(*filtered_lists)

        # convert to lists
        pred_lines_c = list(pred_lines_c)
        norm_pred_lines_c = list(norm_pred_lines_c)
    else:
        pred_lines_c = []
        norm_pred_lines_c = []

    return gt_lines_c, norm_gt_lines_c, gt_cat_list_c, pred_lines_c, norm_pred_lines_c, gt_mix, pred_dataset_mix


def match_gt2pred_simple(gt_items, pred_items, line_type, img_name):

    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines, gt_items, pred_items = get_gt_pred_lines(gt_items, pred_items,line_type)
    match_list = []

    if not norm_gt_lines: # not matched pred should be concatenate
        pred_idx_list = range(len(norm_pred_lines))
        match_list.append({
            'gt_idx': [""],
            'gt': "",
            'pred_idx': pred_idx_list,
            'pred': ''.join(pred_lines[_] for _ in pred_idx_list), 
            'gt_position': [""],
            'pred_position': pred_items[pred_idx_list[0]]['position'][0],  # get the first pred's position
            'norm_gt': "",
            'norm_pred': ''.join(norm_pred_lines[_] for _ in pred_idx_list),
            'gt_category_type': "",
            'pred_category_type': get_pred_category_type(pred_idx_list[0], pred_items), # get the first pred's category
            'gt_attribute': [{}],
            'edit': 1,
            'img_id': img_name
        })
        return match_list,None
    elif not norm_pred_lines: # not matched gt should be separated
        for gt_idx in range(len(norm_gt_lines)):
            match_list.append({
                'gt_idx': [gt_idx],
                'gt': gt_lines[gt_idx],
                'pred_idx': [""],
                'pred': "",
                'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
                'pred_position': "",
                'norm_gt': norm_gt_lines[gt_idx],
                'norm_pred': "",
                'gt_category_type': gt_cat_list[gt_idx],
                'pred_category_type': "",
                'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
                'edit': 1,
                'img_id': img_name
            })
        return match_list,None
    
    cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, norm_pred_lines)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    
    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx in row_ind:
            row_i = list(row_ind).index(gt_idx)
            pred_idx = int(col_ind[row_i])
            pred_line = pred_lines[pred_idx]
            norm_pred_line = norm_pred_lines[pred_idx]
            edit = cost_matrix[gt_idx][pred_idx]
        else:
            pred_idx = ""
            pred_line = ""
            norm_pred_line = ""
            edit = 1
        

        match_list.append({
            'gt_idx': [gt_idx],
            'gt': gt_lines[gt_idx],
            'norm_gt': norm_gt_lines[gt_idx],
            'gt_category_type': gt_cat_list[gt_idx],
            'gt_position': [gt_items[gt_idx].get('order') if gt_items[gt_idx].get('order') else gt_items[gt_idx].get('position', [""])[0]],
            'gt_attribute': [gt_items[gt_idx].get("attribute", {})],
            'pred_idx': [pred_idx],
            'pred': pred_line,
            'norm_pred': norm_pred_line,
            'pred_category_type': get_pred_category_type(pred_idx, pred_items) if pred_idx else "",
            'pred_position': pred_items[pred_idx]['position'][0] if pred_idx else "",
            'edit': edit,
            'img_id': img_name
        })
    
    pred_idx_list = [pred_idx for pred_idx in range(len(norm_pred_lines)) if pred_idx not in col_ind] # get not matched preds
    if pred_idx_list:
        if line_type in ['html_table', 'latex_table']:
            unmatch_table_pred = []
            for i in pred_idx_list:
                original_item = pred_items[i]
                soup = BeautifulSoup(original_item.get('content'),'html.parser')
                text_block = [re.sub(r'\$\\cdot\$','',item.string).strip() for item in soup.findAll('td') if item.string]
                for concatenate_text in text_block:
                    new_item = deepcopy(original_item)
                    new_item['content'] = concatenate_text
                    new_item['category_type'] = 'text_all'  
                    unmatch_table_pred.append(new_item)
            return match_list, unmatch_table_pred  
        
        else:
            match_list.append({
                'gt_idx': [""],
                'gt': "",
                'pred_idx': pred_idx_list,
                'pred': ''.join(pred_lines[_] for _ in pred_idx_list), 
                'gt_position': [""],
                'pred_position': pred_items[pred_idx_list[0]]['position'][0],  # get the first pred's position
                'norm_gt': "",
                'norm_pred': ''.join(norm_pred_lines[_] for _ in pred_idx_list),
                'gt_category_type': "",
                'pred_category_type': get_pred_category_type(pred_idx_list[0], pred_items), # get the first pred's category
                'gt_attribute': [{}],
                'edit': 1,
                'img_id': img_name
            })
    return match_list,None


def match_gt2pred_no_split(gt_items, pred_items, line_type, img_name):
    # directly concatenate gt and pred by position
    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines = get_gt_pred_lines(gt_items, pred_items, line_type)
    gt_line_with_position = []
    for gt_line, norm_gt_line, gt_item in zip(gt_lines, norm_gt_lines, gt_items):
        gt_position = gt_item['order'] if gt_item.get('order') else gt_item.get('position', [""])[0]
        if gt_position:
            gt_line_with_position.append((gt_position, gt_line, norm_gt_line))
    sorted_gt_lines = sorted(gt_line_with_position, key=lambda x: x[0])
    gt = '\n\n'.join([_[1] for _ in sorted_gt_lines])
    norm_gt = '\n\n'.join([_[2] for _ in sorted_gt_lines])
    pred_line_with_position = [(pred_item['position'], pred_line, pred_norm_line) for pred_line, pred_norm_line, pred_item in zip(pred_lines, norm_pred_lines, pred_items)]
    sorted_pred_lines = sorted(pred_line_with_position, key=lambda x: x[0])
    pred = '\n\n'.join([_[1] for _ in sorted_pred_lines])
    norm_pred = '\n\n'.join([_[2] for _ in sorted_pred_lines])
    # edit = Levenshtein.distance(norm_gt, norm_pred)/max(len(norm_gt), len(norm_pred))
    if norm_gt or norm_pred:
        return [{
                'gt_idx': [0],
                'gt': gt,
                'norm_gt': norm_gt,
                'gt_category_type': "text_merge",
                'gt_position': [""],
                'gt_attribute': [{}],
                'pred_idx': [0],
                'pred': pred,
                'norm_pred': norm_pred,
                'pred_category_type': "text_merge",
                'pred_position': "",
                # 'edit': edit,
                'img_id': img_name
            }]
    else:
        return []

def is_all_l(spec: str) -> bool:
    spec = re.sub(r'\s+|\|', '', spec)
    spec = re.sub(r'@{[^}]*}', '', spec)
    spec = re.sub(r'!{[^}]*}', '', spec)
    return bool(spec) and len(spec) == 1 and spec in {'l', 'c', 'r'}

def split_gt_equation_arrays(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split GT dictionary entries with \ \ begin {array}... \ \ end {array}.
        -Only applicable to category_type=='Equation_isolvated 'and latex containing an array.
        -Extract a new entry for each line of formula:
    *Update 'latex'
    *If there is a linew_ith_stans, synchronously replace its internal latex
    *The 'order' consists of 7-->7.1, 7.2
    """
    output = []

    for item in data:
        # Only process dictionaries that meet the conditions
        if (item.get("category_type") == "equation_isolated" and
                "\\begin{array" in item.get("latex", "")):

            # Extract the internal content of the array
            match = ARRAY_RE.search(item["latex"])
            if match:

                spec = match.group("spec")
                if not is_all_l(spec):
                    # If there are mixed elements such as r/c/p {...} in the column, keep the original entry directly
                    output.append(item)
                    continue
                
                body  = match.group("body")
                # Split by LaTeX line separator
                lines = [ln.strip() for ln in re.split(r'\\\\', body) if ln.strip()]

                base_order = float(item["order"])  # 7 -> 7.0，Compatible with float/int

                for idx, line in enumerate(lines, start=1):
                    new_item = deepcopy(item)
                    new_item["latex"] = f"\\[{line}\\]"
                    new_item["order"] = round(base_order + idx / 10, 1)
                    output.append(new_item)
                continue
        # Do not modify other situations
        output.append(item)
    return output

def sort_by_position_skip_inline(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort by position [0] from small to large first;
    If fine_category_type=='Equation_inline ', then place it uniformly at the end,
    And maintain their relative order in the original list (stable sorting).
    """
    # Enumerate preserves the original sequential index for stability when equation_inline is in parallel
    return sorted(
        enumerate(items),
        key=lambda pair: (
            pair[1].get('fine_category_type') == 'equation_inline',  # False < True
            pair[1]['position'][0],                                   # pos start
            pair[0]                                                   # original id
        )
    )

def _wrap(line: str) -> str:
    """Re package single line formula \ \ [... \ \]"""
    return f"\\[{line.strip()}\\]"

def split_equation_arrays(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processing entries with category_type=='Equation_isolvated 'and containing \ \ begin {array}...:
    *Split multi line formula
    *Repackaging content
    ** * Recalculate position/positions**
    """
    out: List[Dict[str, Any]] = []

    for item in data:
        if (item.get("category_type").lower() == "equation_isolated" and
                "\\begin{array" in item.get("content", "")):

            content = item["content"]
            m = ARRAY_RE.search(content)
            if not m:
                out.append(item)
                continue

            if not is_all_l(m.group('spec')):
                out.append(item)
                continue
            
            body = m.group('body')
            lines = [ln.strip() for ln in re.split(r'\\\\', body) if ln.strip()]

            # Global starting character index
            pos_key = "position" if "position" in item else "positions"
            global_start = item[pos_key][0]
            body_start_in_content = m.start('body')
            search_from = 0  # cur pos in body 
            for ln in lines:
                # Find the offset of the current row in the body
                idx_in_body = body.find(ln, search_from)
                if idx_in_body == -1:
                    # Unlikely to happen; Conservative treatment
                    idx_in_body = search_from
                search_from = idx_in_body + len(ln)  # update cur pos

                # calculate global index
                line_start_global = global_start + body_start_in_content + idx_in_body
                line_end_global   = line_start_global + len(ln) - 1

                new_item = deepcopy(item)
                new_item["content"] = _wrap(ln)
                new_item[pos_key]   = [line_start_global, line_end_global]

                out.append(new_item)
            continue
        out.append(item)

    return out

def merge_matches(matches, matching_dict):
    final_matches = {}
    processed_gt_indices = set() 

    for gt_idx, match_info in matches.items():
        pred_indices = match_info['pred_indices']
        edit_distance = match_info['edit_distance']

        pred_key = tuple(sorted(pred_indices))

        if pred_key in final_matches:
            if gt_idx not in processed_gt_indices:
                final_matches[pred_key]['gt_indices'].append(gt_idx)
                processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                'gt_indices': [gt_idx],
                'edit_distance': edit_distance
            }
            processed_gt_indices.add(gt_idx)

    for pred_idx, gt_indices in matching_dict.items():
        pred_key = (pred_idx,) if not isinstance(pred_idx, (list, tuple)) else tuple(sorted(pred_idx))

        if pred_key in final_matches:
            for gt_idx in gt_indices:
                if gt_idx not in processed_gt_indices:
                    final_matches[pred_key]['gt_indices'].append(gt_idx)
                    processed_gt_indices.add(gt_idx)
        else:
            final_matches[pred_key] = {
                'gt_indices': [gt_idx for gt_idx in gt_indices if gt_idx not in processed_gt_indices],
                'edit_distance': None
            }
            processed_gt_indices.update(final_matches[pred_key]['gt_indices'])

    return final_matches