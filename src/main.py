import argparse
import copy
import csv
import json
import math
import multiprocessing
import os
import pickle
import random
import shutil
import sys
import time

from data_processor import DataLoader as dl
from data_processor import Matcher as matcher
from evaluator.ColumnMatcherEval import ColumnMatcherEval
from evaluator.RowMatcher import RowMatcherEval, RowMatcherUnit
from autojoin import autojoin as aj
from evaluator.TransformationSetEval import TransformationSetEval
from pattern import Finder

from Transformation.Blocks.LiteralPatternBlock import LiteralPatternBlock
from Transformation.Blocks.PositionPatternBlock import PositionPatternBlock
from Transformation.Blocks.TokenPatternBlock import TokenPatternBlock
from Transformation.Blocks.SplitSubstrPatternBlock import SplitSubstrPatternBlock
from Transformation.Blocks.TwoCharSplitSubstrPatternBlock import TwoCharSplitSubstrPatternBlock
from Transformation.Blocks.SubstrG import SubstrG
from Transformation.Blocks.SplitG import SplitG
from Transformation.Blocks.SplitSubstrG import SplitSubstrG
from Transformation.Pattern import Pattern

import pathlib

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.absolute())

METHOD = "PT"

# MUST Be Kept updated in order for config file to work. It also should be updated in the Finder.py
PT_PARAMS = {
    'max_tokens': 3,  # maximum number of allowed placeholders
    'max_blocks': 3,  # maximum number of allowed blocks (either placeholder or literal)
    'generalize': False,
    'sample_size': None,
    # 'sample_size': ['path1:data/_samples/AJ/', 'path2:data/_samples/AJ/'],
    # 'sample_size': ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', 'kmeans'],

    'token_splitters': [' ', ],
    # set to None to disable. Break placeholders into other placeholders based on these chars
    'remove_duplicate_patterns': True,
    # After generating all possible transformation, delete the duplicates, may take time
    'switch_literals_placeholders': True,  # Replace placeholder with literals and add them as new pattern
    'only_first_match': False,  # Take only first match for the placeholder or look for all of possible matches.

    'units_to_extract': [LiteralPatternBlock, PositionPatternBlock, TokenPatternBlock, SplitSubstrPatternBlock],
    # 'units_to_extract': [LiteralPatternBlock, SubstrG, SplitG, SplitSubstrG],
    # literal must be included
    # 'units_to_extract': [LiteralPatternBlock, PositionPatternBlock, TokenPatternBlock, SplitSubstrPatternBlock, TwoCharSplitSubstrPatternBlock],  # not including literal

    'prioritize_equal_coverage_patterns': True,

    'extended_transformations': False,

    'extension_mode': 'duplicating',
    'extension_index': 2,

    'src_tgt_swapping_sample_percentage': None,
    'src_tgt_swapping_sample_min_size': 5,

}

GOLDEN_ROWS = True
SWAP_SRC_TARGET = True

DATASET = "BM"
DS_PATH = BASE_PATH + '/data/autojoin-Benchmark/'
# DS_PATH = BASE_PATH + '/data/autojoin-small/'
# DS_PATH = BASE_PATH + '/data/autojoin-no-gt/' # must set GOLDEN_ROW to False
# DS_PATH = BASE_PATH + '/data/synthesis/'
# DATASET = "FF"
# DS_PATH = BASE_PATH + '/data/FlashFill/'


MULTI_CORE = True
NUM_PROCESSORS = 0  # 0: multiprocessing.cpu_count()//2

DO_REMAINING = False
OVERRIDE = True
OUTPUT_PATH = BASE_PATH + "/output/"



MAKE_LOWER=True


CLUSTER_CACHE_PATH = BASE_PATH + "/cache/clustering"

AJ_SUBSET_SIZE = 2
AJ_NUM_SUBSET = 8  # or '5%'

ROW_MATCHING_N_START = 4
ROW_MATCHING_N_END = 20

CNT_CUR = multiprocessing.Value('i', 0)
CNT_ALL = 0


def get_rows_by_sampler(method, rows):
    import cluster_sampling
    import pandas as pd


    res = [(k, list(v)[0]) for k, v in rows.items()]
    df = pd.DataFrame(res, columns=['src', 'target'])

    seperators = cluster_sampling.find_punctuations(df) + [" "]
    df['similarity'] = df.apply(
        lambda x: cluster_sampling.similarity_generator(seperators, x[0], x[1]), axis=1)

    km, df, mat = cluster_sampling.kmeans(df, log=False)

    if method == "kmeans":
        temp = cluster_sampling.cluster_sampling(km, df, mat)
    # elif method == "random":
    #     tmp = cluster_sampling.uniform_sampling(km, df, mat)
    else:
        raise Exception("Wrong sampling method")

    # t = df.loc[0]

    sample = {df.loc[val[0]].src: {df.loc[val[0]].target} for val in temp}
    weights = {df.loc[val[0]].src: val[1] for val in temp}

    return sample, weights


def get_pattern(func, params, verbose=False):
    lst = []
    '''
    You can insert list of some tables in lst to limit loading to those tables, if lst=[], all tables are loaded
    e.g.:
    # lst = [
    #     'synthesis-50rows',
    #     'synthesis-500rows',
    #     'synthesis-5000rows',
    # ]
    '''

    # lst = ['park to state 2', 'chinese provinces', 'us presidents 5']

    tables, all_tables = dl.get_tables_from_dir(DS_PATH, lst, make_lower=MAKE_LOWER, verbose=False)

    print("Reading Done!")

    res = matcher.get_matching_tables_golden(tables, bidi=False)

    ''' This Part is for experimental column matching, still not complete and should be commented:

    # match_start_time = time.time()
    # res = matcher.get_matching_tables(all_tables, tables, q=6, matching_ratio=0.7,
    #                                   src_keys_ratio=0.2, tgt_keys_ratio=0.2, verbose=True)
    # match_end_time = time.time()
    # print("Matcher RunTime: --- %f seconds ---" % (match_end_time - match_start_time))
    #
    # cme = ColumnMatcherEval(all_tables, tables, res)
    # print(cme)

    # item = res['items'][1]
    # rows, is_swapped = matcher.get_matching_rows_golden(tables, item, swap_src_target=SWAP_SRC_TARGET)
    # Finder.get_patterns(rows, max_tokens=3)
    '''
    done_tbl = []
    if DO_REMAINING and os.path.exists(OUTPUT_FILE):
        skip = 0
        items_new = []

        with open(OUTPUT_FILE, "r") as f:
            for line in f.readlines():
                tab = line.strip().split(',')[0]
                if tab != "file_name":
                    done_tbl.append(tab)

        for item in res['items']:
            tbl_name = item['src_table'][4:]
            if tbl_name in done_tbl:
                assert os.path.exists(OUTPUT_DIR + tbl_name + ".txt")
                skip += 1
            else:
                items_new.append(item)

        print(f"{skip}/{len(res['items'])} already done.")
        res['items'] = items_new

    all_rows = []
    data = []

    if not MULTI_CORE:
        print("Running on single core mode")

    global CNT_ALL
    CNT_ALL = len(res['items'])

    all_has_gt = True

    skipped_tbls = 0
    total_skip_cnt = 0

    for item in res['items']:

        has_GT = 'GT' in tables[item['src_table'][4:]]
        all_has_gt = all_has_gt and has_GT

        force_swap = 0
        src_tgt_swapping_start_time = time.time()

        if params['src_tgt_swapping_sample_percentage'] is None:
            print(f"Matching rows for '{item['src_table']}' ...")
            src_tgt_swapping_time = 0
            force_swap = 0
            golden_force_swap = "UNK"
        else:
            assert func == 'run_pattern'
            print(f"Finding direction of rows for '{item['src_table']}' ...")

            assert GOLDEN_ROWS

            rows_org, _ = matcher.get_matching_rows_golden(tables, item, swap_src_target=False, force_swap=-1)
            rows_swp, _ = matcher.get_matching_rows_golden(tables, item, swap_src_target=True, force_swap=1)

            # if len(rows_org) != len(rows_swp):
            #     print(f"org len: {len(rows_org)},swp len: {len(rows_swp)}")
            #     raise AssertionError

            len_rows = len(rows_org)
            dir_sample_size = max(math.floor(len_rows*params['src_tgt_swapping_sample_percentage']/100),
                                  params['src_tgt_swapping_sample_min_size'])
            dir_sample_size = min(dir_sample_size, len_rows)

            keys = random.sample(list(rows_org), dir_sample_size)
            sample_rows_org = {}
            for k in keys:
                sample_rows_org[k] = rows_org[k]

            extras = {}
            if PT_PARAMS['extended_transformations']:
                extras = {
                    "file_path": DS_PATH + f"/{item['src_table'][4:]}/ground truth.csv",
                    "src": f"source-{item['src_row']}",
                    "target": f"target-{item['target_row']}",
                    'extension_mode': params['extension_mode'],
                    'extension_index': params['extension_index'],
                }

            org_res = Finder.get_patterns(sample_rows_org, params=params, table_name=None, extras=extras, verbose=False)

            len_rows = len(rows_swp)
            dir_sample_size = max(math.floor(len_rows * params['src_tgt_swapping_sample_percentage'] / 100),
                                  params['src_tgt_swapping_sample_min_size'])
            dir_sample_size = min(dir_sample_size, len_rows)

            keys = random.sample(list(rows_swp), dir_sample_size)
            sample_rows_swp = {}
            for k in keys:
                sample_rows_swp[k] = rows_swp[k]

            swp_res = Finder.get_patterns(sample_rows_swp, params=params, table_name=None, extras=extras, verbose=False)


            tr_org = len(org_res['ranked'])
            tr_swp = len(swp_res['ranked'])

            force_swap = 1 if tr_swp < tr_org else -1

            src_tgt_swapping_time = time.time() - src_tgt_swapping_start_time


            print("Matching golden set for finding src and target")
            params_temp = copy.deepcopy(params)
            params_temp['extended_transformations'] = False
            org_res_gld = Finder.get_patterns(rows_org, params=params_temp, table_name=None, extras={}, verbose=False)
            swp_res_gld = Finder.get_patterns(rows_swp, params=params_temp, table_name=None, extras={}, verbose=False)
            # golden_force_swap = 1 if len(swp_res_gld['ranked']) < len(org_res_gld['ranked']) else -1
            if len(swp_res_gld['ranked']) < len(org_res_gld['ranked']):
                golden_force_swap = 1
            elif len(swp_res_gld['ranked']) == len(org_res_gld['ranked']):
                golden_force_swap = 2
            else:
                golden_force_swap = -1



        params_copy = copy.deepcopy(params)
        params_copy['src_tgt_swapping_time'] = src_tgt_swapping_time
        params_copy['force_swap'] = force_swap
        params_copy['golden_force_swap'] = golden_force_swap

        params_copy['estimated_swap'] = "UNK"



        if GOLDEN_ROWS:
            if not has_GT:
                raise Exception(
                    'golden row matching cannot be used when no ground truth table exists for ' + item['src_table'][4:])

            rows, is_swapped = matcher.get_matching_rows_golden(tables, item, swap_src_target=SWAP_SRC_TARGET, force_swap=force_swap)
        else:
            rows, is_swapped = matcher.get_matching_rows_for_table(tables, item, ROW_MATCHING_N_START, ROW_MATCHING_N_END,
                                                         swap_src_target=SWAP_SRC_TARGET, force_swap=force_swap)
        if verbose:
            print(f"source and target columns are {'' if is_swapped else 'NOT '}swapped")
        new_rows = {}
        for src, target in rows.items():
            new_rows[src] = [[t, t] for t in target]


        if force_swap == 0:
            params_copy['estimated_swap'] = 1 if is_swapped else -1
        elif has_GT:
            _, tmp_is_swapped = matcher.get_matching_rows_golden(tables, item, swap_src_target=True, force_swap=0)
            params_copy['estimated_swap'] = 1 if tmp_is_swapped else -1

        rr = {
            'col_info': item,
            'is_swapped': is_swapped,
            'rows': new_rows
        }
        all_rows.append(rr)

        params_copy['is_swapped'] = is_swapped

        rmu = None
        if has_GT:
            rmu = RowMatcherUnit(tables, rr)
            print(rmu)



        if func == 'run_pattern' and params['sample_size'] is not None:
            cached_sampling_size = 0
            db_cache_dir = os.path.basename(os.path.normpath(DS_PATH))
            cache_file_path = CLUSTER_CACHE_PATH + f"/{db_cache_dir}/" + item['src_table'] + ".pkl"
            subset_size = params['sample_size']
            if not isinstance(subset_size, (list, tuple, set,)):
                subset_size = [subset_size]

            CNT_ALL += len(subset_size) - 1

            for size in subset_size:
                sampling_start_time = time.time()
                size_txt = size
                sample_tbl_name = item['src_table'][4:] + "_s-" + str(size_txt).split(":")[0]

                total_skip_cnt += 1
                if DO_REMAINING:
                    if sample_tbl_name in done_tbl:
                        assert os.path.exists(OUTPUT_DIR + sample_tbl_name + ".txt")
                        skipped_tbls += 1
                        print(f"{sample_tbl_name} already done. Skipping!")
                        continue

                if type(size) == str:
                    if size[-1] == '%':
                        percent = int(size[:-1])
                        a = float(percent) / 100
                        size = int(a * len(rows)) + 1
                        if size > len(rows):
                            size = len(rows)
                        random_sample = True

                    elif size.startswith("path"):
                        random_sample = False
                        # {'src1': {'target1'}}
                        file_name = f"{item['src_table'][4:]}.csv"
                        file_path = pathlib.Path(BASE_PATH) / size.split(":", 1)[1] / file_name
                        sample_rows = {}
                        with open(file_path, 'r') as fs:
                            reader = csv.reader(fs, quotechar=None)
                            head = next(reader)
                            assert head[0].lower() == "source" and head[1].lower() == "target"
                            for line in reader:
                                if MAKE_LOWER:
                                    sam_s, sam_t = line[0].lower(), line[1].lower()
                                else:
                                    sam_s, sam_t = line[0], line[1]

                                if is_swapped:
                                    sam_s, sam_t = sam_t, sam_s

                                sample_rows[sam_s] = {sam_t}

                                if sam_s not in rows:
                                    print(f"WARNING: {size.split(':')[0]} FOR TABLE {sample_tbl_name}: value {sam_s} NOT FOUND!")

                    else:
                        random_sample = False
                        if os.path.exists(cache_file_path):
                            print("Loading clusters from cache")
                            with open(cache_file_path, 'rb') as fc:
                                cdata = pickle.load(fc)
                            sample_rows = cdata['sample_rows']
                            weights = cdata['weights']
                            cached_sampling_size = cdata['sampling_time']
                        else:
                            sample_rows, weights = get_rows_by_sampler(size, rows)

                else:
                    size = int(size)


                p2 = copy.deepcopy(params_copy)



                if random_sample:
                    keys = random.sample(list(rows), size)
                    sample_rows = {}
                    for k in keys:
                        sample_rows[k] = rows[k]
                    weights = None
                else:
                    if not size.startswith("path"):
                        p2['sampling_weights'] = weights
                    assert len(sample_rows) > 0


                if cached_sampling_size == 0:
                    p2['sampling_time'] = time.time() - sampling_start_time
                else:
                    p2['sampling_time'] = cached_sampling_size


                if cached_sampling_size == 0 and not random_sample and not size.startswith("path"):
                    if not os.path.exists(CLUSTER_CACHE_PATH+f"/{db_cache_dir}"):
                        pathlib.Path(CLUSTER_CACHE_PATH+f"/{db_cache_dir}").mkdir(parents=True, exist_ok=True)

                    print(f"Saving {cache_file_path}")

                    with open(cache_file_path, 'wb') as fc:
                        pickle.dump({
                            'sample_rows': sample_rows,
                            'weights': weights,
                            'sampling_time': p2['sampling_time'],
                        }, fc)


                if MULTI_CORE:
                    data.append((item, sample_rows, p2, tables, sample_tbl_name, rmu, verbose))
                else:
                    globals()[func](item, sample_rows, p2, tables, sample_tbl_name, rmu, verbose=verbose)


        else:
            if MULTI_CORE:
                data.append((item, rows, params_copy, tables, None, rmu, verbose))
            else:
                globals()[func](item, rows, params_copy, tables, None, rmu, verbose=verbose)

    CNT_ALL -= skipped_tbls

    print(f"{skipped_tbls} / {total_skip_cnt} sampled tables skipped, {CNT_ALL} remaining")


    if all_has_gt and len(all_rows) > 0:
        rme = RowMatcherEval(tables, all_rows)
        print("Row matching performance:" + str(rme))

    if MULTI_CORE:
        print(f"Using {NUM_PROCESSORS} processes...")

        # @TODO: Support spawn
        if sys.platform in ('win32', 'msys', 'cygwin'):
            print("fork based multi core processing works only on *NIX type operating systems.")
            sys.exit(1)

        from multiprocessing import get_context
        pool = get_context('fork').Pool(processes=NUM_PROCESSORS)
        # pool = multiprocessing.Pool(processes=NUM_PROCESSORS)

        rets = pool.starmap(globals()[func], data)
        pool.close()

        pool.join()


def run_pattern(item, rows, params, tables, table_name=None, rmu=None, verbose=None):
    if table_name is None:
        table_name = item['src_table'][4:]

    extras = {}
    if PT_PARAMS['extended_transformations']:
        extras = {
            "file_path": DS_PATH + f"/{item['src_table'][4:]}/ground truth.csv",
            "src": f"source-{item['src_row']}",
            "target": f"target-{item['target_row']}",
            'extension_mode': params['extension_mode'],
            'extension_index': params['extension_index'],
        }

    res = Finder.get_patterns(rows, params=params, table_name=table_name, extras=extras, verbose=verbose)

    if rmu is not None:
        tr_eval = TransformationSetEval(tables, item, [r[2] for r in res['patterns']], SWAP_SRC_TARGET)
    else:
        if verbose:
            print("No golden set provided")
        tr_eval = None

    global CNT_CUR
    with CNT_CUR.get_lock():
        CNT_CUR.value += 1
    print(f"({CNT_CUR.value}/{CNT_ALL}) -> " + table_name)
    print("Total run time: %.2f s" % res['runtime'])
    print(f"{len(res['patterns'])} patterns / {res['input_len']} inputs \n-----------")
    if verbose:
        print(tr_eval)
    pt_print(res, table_name, rmu, tr_eval)
    # pt_print(res, table_name, rmu, None)


def run_aj(item, rows, params, tables, table_name=None, rmu=None, verbose=False):
    # Verbose = limited,yes,full

    if table_name is None:
        table_name = item['src_table'][4:]
    tbl = table_name

    num_subsets = params['num_subsets']
    subset_size = params['subset_size']

    '''
    if isinstance(subset_size, (list, tuple, set,)):
        if PARAM_MULTI_CORE:
            prm = []
            for size in subset_size:
                assert not isinstance(subset_size, (list, tuple, set,))
                prm.append((item, rows, {'subset_size': size, 'num_subsets': num_subsets}, rmu, verbose, tbl + f"_subsize_{size}"))
            # print(f"Using {PARAM_NUM_PROCESSORS} processes for params...")
            pool = multiprocessing.Pool(processes=PARAM_NUM_PROCESSORS)
            rets = pool.starmap(run_aj, prm)
            pool.close()
            pool.join()
            for res in rets:
                print(f": {res['print_name']} Done in %.2f s" % res['runtime'])
                aj_print(res, res['print_name'], rmu)
        else:
            i = 0
            for size in subset_size:
                i += 1
                assert not isinstance(size, (list, tuple, set,))
                res = aj.auto_join(rows, size, num_subsets, verbose=verbose, print_name=tbl + f"_subsize_{size}")
                print(f"Param {i}/{len(subset_size)}: {res['print_name']} Done in %.2f s" % res['runtime'])
                aj_print(res, res['print_name'], rmu) 
    else:
        res = aj.auto_join(rows, subset_size, num_subsets, verbose=verbose, print_name=tbl)
        print(f"{res['print_name']}")
        print("Total time: %.2f s" % res['full_time'] + ", Total spent time: %.2f s" % res['runtime'])
        print("---------")
        aj_print(res, res['print_name'], rmu)


'''

    if type(subset_size) == str:
        assert subset_size[-1] == '%'
        percent = int(subset_size[:-1])
        a = float(percent) / 100
        subset_size = int(a * len(rows)) + 1
        if subset_size > len(rows):
            subset_size = len(rows)
        print(f"{tbl} -> Subset size: {percent}% = {subset_size}")

    res = aj.auto_join(rows, subset_size, num_subsets, verbose=verbose, print_name=tbl)

    if rmu is not None:
        tr_eval = TransformationSetEval(tables, item, [r[0] for r in res['ranked']], SWAP_SRC_TARGET)
    else:
        if verbose:
            print("No golden set provided")
        tr_eval = None

    global CNT_CUR
    with CNT_CUR.get_lock():
        CNT_CUR.value += 1
    print(f"({CNT_CUR.value}/{CNT_ALL}) -> {res['print_name']}")
    print("Total time: %.2f s" % res['full_time'] + ", Total spent time: %.2f s" % res['runtime'])
    print("---------")
    if verbose:
        print(tr_eval)
    aj_print(res, res['print_name'], rmu, tr_eval)


def pt_print(res, filename, row_matcher, tr_eval):
    coverage = res['covered'] / res['input_len'] if res['input_len'] != 0 else "NA"
    best_coverage = res['ranked'][0][1] / res['input_len'] if len(res['ranked']) > 0 else math.nan
    best_coverage2 = res['ranked'][1][1] / res['input_len'] if len(res['ranked']) > 1 else math.nan
    best_coverage3 = res['ranked'][2][1] / res['input_len'] if len(res['ranked']) > 2 else math.nan

    total_patterns = len(res['ranked'])
    sampling_time = f"{res['params']['sampling_time']:.10f}" if 'sampling_time' in res['params'] else 0
    direction_time = f"{res['params']['src_tgt_swapping_time']:.10f}" if 'src_tgt_swapping_time' in res['params'] else 0
    force_swap = str(res['params']['force_swap']) if 'force_swap' in res['params'] else 0

    sampling_weights = res['params'].get('sampling_weights', None)

    # try:
    #     input_rows = len(tr_eval.inp_pat)
    # except AttributeError:
    #     tr_eval = None

    input_rows = len(tr_eval.inp_pat) if tr_eval is not None else "NA"

    sum_units = sum(len(tr[2]) if type(tr[2]) is Pattern else math.nan for tr in res['ranked'])
    avg_units = sum_units / total_patterns

    sum_params = sum(tr[2].get_param_count() if type(tr[2]) is Pattern else math.nan for tr in res['ranked'])
    avg_params = sum_params / total_patterns

    if not os.path.exists(OUTPUT_FILE):
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FILE, "a+") as f:
            print(
                "file_name,Input Rows,avg_len_input,avg_len_output,Matched pairs,Total Patterns," +
                "Golden Coverage,Golden Best tran Coverage,Golden second tran Coverage,Golden third tran Coverage,Coverage,Best tran Coverage,Second tran Coverage,Third tran Coverage," +
                "Runtime,Sampling time(NOT In RUNTIME),Direction time(NOT In RUNTIME),Extending time,Effective Generalizations,Generalization Time," +
                "force_swap,estimated_swap,golden_swap,Avg unit per trans.,Sum unit per trans.,Avg param per trans.,Sum param per trans.," +
                "Row_Match_P,Row_Match_R,Row_Match_F1," +
                "all_generated_placeholder_comb,all_removed_placeholder_comb,all_remaining_placeholder_comb,init_remaining_placeholder_comb,splitted_remaining_placeholder_comb," +
                "placeholder_gen_time,extract_pat_time,duplicate_pat_remove_time,pat_applying_time,get_covering_set_time," +
                "Num placeholder comb,Num All Patterns,Num duplicate patterns removed,Num patterns to try," +
                "cnt_all_patterns_all_rows,cnt_patterns_successful,cnt_patterns_failed,cnt_patterns_hit_cache"
                , file=f)

    with open(OUTPUT_FILE, "a+") as f:
        rows = f"{row_matcher.precision},{row_matcher.recall},{row_matcher.f1}" if row_matcher is not None else "-1,-1,-1"
        tr_eval_txt = f"{tr_eval.coverage},{tr_eval.best_pattern_coverage},{tr_eval.kth_pattern_coverage(2)},{tr_eval.kth_pattern_coverage(3)}," if tr_eval is not None else "NA,NA,NA,NA"
        print(
            f"{filename},{input_rows},{res['avg_len_input']},{res['avg_len_output']},{res['input_len']},{total_patterns}," +
            tr_eval_txt +
            f"{coverage},{best_coverage},{best_coverage2},{best_coverage3}," +
            f"{res['runtime']},{sampling_time},{direction_time},{res['extending_time']},{res['effective_gens']},{res['gen_time']}," +
            f"{force_swap},{res['params']['estimated_swap']},{res['params']['golden_force_swap']},{avg_units},{sum_units},{avg_params},{sum_params}," +
            rows + "," +
            f"{res['cnt_all_generated_placeholder_comb']},{res['cnt_all_removed_placeholder_comb']},{res['cnt_all_remaining_placeholder_comb']},{res['cnt_init_remaining_placeholder_comb']},{res['cnt_splitted_remaining_placeholder_comb']}," +
            f"{res['placeholder_gen_time']},{res['extract_pat_time']},{res['duplicate_pat_remove_time']},{res['pat_applying_time']},{res['get_covering_set_time']}," +
            f"{res['cnt_placeholder_comb']},{res['cnt_all_patterns']},{res['cnt_dup_patterns_removed']},{res['cnt_pattern_remaining']}," +
            f"{res['cnt_all_patterns_all_rows']},{res['cnt_patterns_successful']},{res['cnt_patterns_failed']},{res['cnt_patterns_hit_cache']}"
            , file=f)

    with open(OUTPUT_DIR + filename + ".txt", "a+") as f:
        print("---------------", file=f)
        print(f"Results for {filename}", file=f)
        if tr_eval is not None:
            print(f"Coverage in golden matched rows: {tr_eval.covered}/{len(tr_eval.inp_pat)} = %.2f%%" % (
                        tr_eval.covered * 100 / len(tr_eval.inp_pat)), file=f)
            print(f"Best pattern Coverage  in golden matched rows: %.2f%%" % (tr_eval.best_pattern_coverage * 100),
                  file=f)

            print(f"Second pattern Coverage  in golden matched rows: %.2f%%" % (tr_eval.kth_pattern_coverage(2) * 100),
                  file=f)

            print(f"Third pattern Coverage  in golden matched rows: %.2f%%" % (tr_eval.kth_pattern_coverage(3) * 100),
                  file=f)

        if res['input_len'] != 0:
            print(f"Coverage: {res['covered']}/{res['input_len']} = %.2f%%" % (res['covered'] * 100 / res['input_len']),
                  file=f)
        else:
            print(f"Coverage: NA", file=f)

        print(f"Best pattern Coverage: %.2f%%" % (best_coverage * 100), file=f)
        print(f"Second Best pattern Coverage: %.2f%%" % (best_coverage2 * 100), file=f)
        print(f"Third Best pattern Coverage: %.2f%%" % (best_coverage3 * 100), file=f)
        print(f"Total Patterns: {total_patterns}, input rows: {res['input_len']}", file=f)
        print(
            f"Average Input length (Num chars): {res['avg_len_input']}, Average output length (Num chars): {res['avg_len_output']}",
            file=f)
        print(f"Max Tokens: {res['max_tokens']}", file=f)
        print(f"Total Run time: {res['runtime']}, Generalization time:{res['gen_time']}", file=f)
        print(f"Sampling time (NOT ADDED TO RUNTIME): {sampling_time}", file=f)
        print(f"Finding Direction od src and tgt time (NOT ADDED TO RUNTIME): {direction_time}", file=f)
        print(f"Extending time: {res['extending_time']}", file=f)
        print(f"Force src and tgt swapping: {force_swap}", file=f)
        print(f"Estimated src and tgt swapping: {res['params']['estimated_swap']}", file=f)
        print(f"Total number of placeholder combinations (After removing duplicates): {res['cnt_placeholder_comb']}",
              file=f)
        print(f"Total number of all generated patterns: {res['cnt_all_patterns']}", file=f)
        print(f"Number of duplicate patterns removed: {res['cnt_dup_patterns_removed']}", file=f)
        print(f"Total number of generated patterns to try: {res['cnt_pattern_remaining']}", file=f)
        print(f"Effective Generalizations: {res['effective_gens']}", file=f)

        print(f"\nNumber of all generated placeholder comb.: {res['cnt_all_generated_placeholder_comb']}", file=f)
        print(f"Number of all remaining placeholder comb.: {res['cnt_all_remaining_placeholder_comb']}", file=f)
        print(f"Number of all removed placeholder comb.: {res['cnt_all_removed_placeholder_comb']}", file=f)
        print(f"Number of init remaining placeholder comb.: {res['cnt_init_remaining_placeholder_comb']}", file=f)
        print(f"Number of splitted remaining placeholder comb.: {res['cnt_splitted_remaining_placeholder_comb']}",
              file=f)
        print(f"Number of removed init placeholder comb.: {res['cnt_removed_init_placeholder_comb']}", file=f)
        print(f"Number of removed splitted placeholder comb.: {res['cnt_removed_splitted_placeholder_comb']}", file=f)

        print(f"\nplaceholder generation time: {res['placeholder_gen_time']}", file=f)
        print(f"extract patterns time: {res['extract_pat_time']}", file=f)
        print(f"duplicate patterns remove time: {res['duplicate_pat_remove_time']}", file=f)
        print(f"patterns applying time: {res['pat_applying_time']}", file=f)
        print(f"get covering set time: {res['get_covering_set_time']}", file=f)

        print(
            f"\nNumber of all patterns that applied on all rows (#pattern * #rows) (complexity): {res['cnt_all_patterns_all_rows']}",
            file=f)
        print(f"Number of successful patterns: {res['cnt_patterns_successful']}", file=f)
        print(f"Number of failed patterns: {res['cnt_patterns_failed']}", file=f)
        print(f"Number of failed patterns filtered by cache (cache hit): {res['cnt_patterns_hit_cache']}", file=f)

        print(f"\nSum of units per transformation: {sum_units}", file=f)
        print(f"Average units per transformation: {avg_units}", file=f)
        print(f"\nSum of params per transformation: {sum_params}", file=f)
        print(f"Average params per transformation: {avg_params}", file=f)

        s = "\nGolden" if GOLDEN_ROWS else "N-gram"
        if row_matcher is not None:
            print(f"{s} Row Matching: P:{row_matcher.precision},R:{row_matcher.recall},F:{row_matcher.f1}", file=f)
        print(f"Params: {res['params']}", file=f)
        print("pattern list: {", file=f)
        s = ""
        tr_weights = {}
        for tr in res['ranked']:
            s += "                 "
            old_pat = ""
            if len(tr) == 5:
                old_pat = f"  -- Original Trans.: {tr[4]}"
            s += str(tr[2]) + f" -> {tr[1]}/{res['input_len']}{old_pat}\n"
            for inp in tr[3]:
                s += "                     |->" + str(inp) + "\n"
                if sampling_weights is not None:
                    tr_weights[tr[2]] = tr_weights.get(tr[2], 0) + sampling_weights[inp[0]]

        print(s, file=f)

        if tr_eval is not None:
            print("****Golden rows apply patterns******", file=f)
            print("golden pattern list: {", file=f)
            s = ""
            for num, tr in enumerate(tr_eval.transformation_list):
                s += "                 "
                s += str(tr) + f" -> {len(tr_eval.pat_inp[num])}/{len(tr_eval.inp_pat)}\n"
                for inp in tr_eval.pat_inp[num]:
                    s += "                     |->" + str(tr_eval._inputs[inp]) + "\n"

            print(s, file=f)

            print("--- Not covered inputs:", file=f)
            for i, inp in enumerate(tr_eval.inp_pat):
                if len(inp) == 0:
                    print("   " + str(tr_eval._inputs[i]), file=f)


        if res['extend_res'] is not None:
            EXT_FILE = OUTPUT_DIR + "/__exts.csv"

            if not os.path.exists(EXT_FILE):
                with open(EXT_FILE, "a+") as f2:
                    print("table,transformation,matched_pairs,old_cov,new_cov,golden_matched_pairs,golden_old_cov,golden_new_cov,old_weights,new_weights", file=f2)


            print("*******Extends******", file=f)


            with open(EXT_FILE, "a+") as f2:
                for tr in res['extend_res']:
                    golden_old_cov, golden_new_con = math.nan, math.nan
                    if tr_eval is not None:
                        golden_old_cov = tr_eval.get_transformation_coverage(tr['old_tr'])
                        golden_new_con = tr_eval.get_transformation_coverage(tr['transformation'])


                    old_weight = sum(sampling_weights[inp[0]] for inp in tr['old_pat_inp']) if sampling_weights is not None else "NA"
                    new_weight = sum(sampling_weights[inp[0]] for inp in tr['new_pat_inp']) if sampling_weights is not None else "NA"

                    ext_st = f"{filename},{str(tr['transformation']).replace(',',' ')},{res['input_len']},{tr['old_cov']},{tr['new_cov']},"
                    ext_st += f"{input_rows},{golden_old_cov},{golden_new_con},"
                    ext_st += f"{old_weight},{new_weight}"
                    print(ext_st, file=f2)
                    ext_list_str = len(tr['extended_list'])
                    print(
                        f"{tr['old_tr']}:\n    To:{ext_list_str}\n    Cov old:{tr['old_cov']}/ Cov new:{tr['new_cov']}",
                        file=f)
                    each_cover = tr['each_cover']
                    if len(tr['each_cover']) > 10000:
                        print("    * Limiting printed output to first 10,000 extended transformations", file=f)
                        each_cover = copy.deepcopy(tr['each_cover'][:10000])
                    for t in each_cover:
                        k = list(t.keys())[0]
                        print(f"    {k} ({len(t[k])}):", file=f)
                        prints_n = 0
                        for inp in t[k]:
                            if prints_n == 5:
                                print(f"       ->skipping", file=f)
                                break
                            print(f"       -> {inp}", file=f)
                            prints_n += 1

                    print("-------", file=f)


        if len(tr_weights) > 0:
            WEIGHT_FILE = OUTPUT_DIR + "/__weights.csv"

            if not os.path.exists(WEIGHT_FILE):
                with open(WEIGHT_FILE, "a+") as f2:
                    print("table,transformation,matched_pairs,input_rows,estimated weight,real weight,est. cov.,real_coverage", file=f2)

            tr_real_coverage = {}
            if tr_eval is not None:
                for num, tr in enumerate(tr_eval.transformation_list):
                    tr_real_coverage[tr] = len(tr_eval.pat_inp[num])

            input_rows = len(tr_eval.inp_pat) if tr_eval is not None else math.nan


            with open(WEIGHT_FILE, "a+") as f2:
                for ktr, wt in tr_weights.items():
                    cov = tr_real_coverage.get(ktr, math.nan)
                    est_cov = int(wt * input_rows) if tr_eval is not None else "NA"
                    ss = f"{filename},{str(ktr).replace(',',' ')},{res['input_len']},"
                    ss += f"{input_rows},{wt},{cov / input_rows},{est_cov},{cov}"
                    print(ss, file=f2)

        print("****************", file=f)


def aj_print(res, filename, row_matcher, tr_eval):
    best_coverage = res['ranked'][0][1] / res['input_len'] if len(res['ranked']) > 0 else 0
    total_patterns = len(res['ranked'])
    input_rows = len(tr_eval.inp_pat) if tr_eval is not None else "NA"

    if not os.path.exists(OUTPUT_FILE):
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FILE, "a+") as f:
            print(
                "file_name,Input Rows,Matched pairs,Total Patterns," +
                "Golden Coverage,Golden Best tran Coverage,Coverage,Best tran Coverage," +
                "Required Runtime,Execution time,Ranking time," +
                "rows per subset,number of subsets,level threshold,gain_threshold," +
                "Row_Match_P,Row_Match_R,Row_Match_F1",
                file=f)

    with open(OUTPUT_FILE, "a+") as f:
        rows = f"{row_matcher.precision},{row_matcher.recall},{row_matcher.f1}" if row_matcher is not None else "-1,-1,-1"
        tr_eval_txt = f"{tr_eval.coverage},{tr_eval.best_pattern_coverage}," if tr_eval is not None else "NA,NA,"
        print(f"{filename},{input_rows},{res['input_len']},{total_patterns}," +
              tr_eval_txt +
              f"{res['covered'] / res['input_len']},{best_coverage}," +
              f"{res['full_time']},{res['runtime']},{res['rank_time']}," +
              f"{res['subset_size']},{res['num_subsets']},{res['level_threshold']},{res['gain_threshold']}," +
              rows
              , file=f)

    with open(OUTPUT_DIR + filename + ".txt", "a+") as f:
        print("---------------", file=f)
        print(f"Results for {filename}", file=f)
        if tr_eval is not None:
            print(f"Coverage in golden matched rows: {tr_eval.covered}/{len(tr_eval.inp_pat)} = %.2f%%" % (
                        tr_eval.covered * 100 / len(tr_eval.inp_pat)), file=f)
            print(f"Best pattern Coverage  in golden matched rows: %.2f%%" % (tr_eval.best_pattern_coverage * 100),
                  file=f)
        print(f"Coverage: {res['covered']}/{res['input_len']} = %.2f%%" % (res['covered'] * 100 / res['input_len']),
              file=f)
        print(f"Best pattern Coverage: %.2f%%" % (best_coverage * 100), file=f)
        print(f"Total Patterns: {total_patterns}, input rows: {res['input_len']}", file=f)
        print(f"Total required runtime: {res['full_time']}", file=f)
        print(f"Execution time: {res['runtime']}, Ranking time:{res['rank_time']}", file=f)
        print(f"rows per subset: {res['subset_size']}, number of subsets: {res['num_subsets']}", file=f)
        print(f"level threshold: {res['level_threshold']}, gain_threshold: {res['gain_threshold']}", file=f)
        s = "Golden" if GOLDEN_ROWS else "N-gram"
        if row_matcher is not None:
            print(f"{s} Row Matching: P:{row_matcher.precision},R:{row_matcher.recall},F:{row_matcher.f1}", file=f)
        print(f"blocks: {res['blocks']}", file=f)
        print("ranked list: {", file=f)
        s = ""
        for tr in res['ranked']:
            s += "                 "
            s += str(tr[0]) + f" -> {tr[1]}/{res['input_len']}\n"

        print(s, file=f)
        print("             }", file=f)

        print("subset list: {", file=f)
        s = ""
        for tr in res['subset_res']:
            s += "                 {\n"
            s += f"                     subset:{tr['subset']}\n"
            s += f"                     transformation:{tr['transformation']}\n"
            s += f"                     time:{tr['time']}\n"
            s += "                 }\n"

        print(s, file=f)
        print("             }", file=f)
        print("****************", file=f)


def row_matching_test(n_start_from=2, n_start_to=25, file_write=True):
    if file_write:
        if not os.path.exists(OUTPUT_FILE):
            pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "a+") as f:
                print("min_n,p,r,f1", file=f)

    lst = []
    tables, all_tables = dl.get_tables_from_dir(DS_PATH, lst, make_lower=True, verbose=False)
    print("Reading Done!")

    res = matcher.get_matching_tables_golden(tables, bidi=False)

    for n_start in range(n_start_from, n_start_to):
        file_path = OUTPUT_DIR + f"/n_start_{n_start:02d}.csv"
        all_rows = []

        for item in res['items']:
            print(f"Matching rows for '{item['src_table']}' ...")

            rows, is_swapped = matcher.get_matching_rows_for_table(tables, item, n_start, ROW_MATCHING_N_END,
                                                         swap_src_target=SWAP_SRC_TARGET)
            new_rows = {}
            for src, target in rows.items():
                new_rows[src] = [[t, t] for t in target]

            rr = {
                'col_info': item,
                'is_swapped': is_swapped,
                'rows': new_rows
            }
            all_rows.append(rr)

            rmu = RowMatcherUnit(tables, rr)
            print(rmu)
            if file_write:
                if not os.path.exists(file_path):
                    with open(file_path, "a+") as f1:
                        print("table,rows,tp,fp,fn,p,r,f", file=f1)
                with open(file_path, "a+") as f1:
                    print(
                        f"{item['src_table'][4:]},{rmu.tp + rmu.fn},{rmu.tp},{rmu.fp},{rmu.fn},{rmu.precision},{rmu.recall},{rmu.f1}"
                        , file=f1)

        rme = RowMatcherEval(tables, all_rows)
        print("Row matching performance:" + str(rme))
        if file_write:
            with open(OUTPUT_FILE, "a+") as f:
                print(f"{n_start},{rme.precision},{rme.recall},{rme.f1}", file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', action='store', type=str, required=False,
                        default='', help='Path of config file')
    args = parser.parse_args()
    if args.config != '':
        cnf_path = str(pathlib.Path(args.config).absolute())
        print(f"Loading config file: {cnf_path}")
        with open(cnf_path, "r") as f:
            cnf = json.load(f)
        print(cnf)
        MULTI_CORE = cnf.get('multicore', MULTI_CORE)
        NUM_PROCESSORS = cnf.get('num_processors', NUM_PROCESSORS)
        GOLDEN_ROWS = cnf.get('golden_rows', GOLDEN_ROWS)
        DO_REMAINING = cnf.get('do_remaining', DO_REMAINING)
        OVERRIDE = cnf.get('override', OVERRIDE)
        DS_PATH = cnf.get('dataset_path', DS_PATH)
        OUTPUT_PATH = cnf.get('output_path', OUTPUT_PATH)
        if cnf.get('add_base_path', True):
            DS_PATH = BASE_PATH + DS_PATH
            OUTPUT_PATH = BASE_PATH + OUTPUT_PATH
        DATASET = cnf.get('dataset', DATASET)
        METHOD = cnf.get('method', METHOD)
        AJ_NUM_SUBSET = cnf.get('aj_num_subset', AJ_NUM_SUBSET)
        AJ_SUBSET_SIZE = cnf.get('aj_subset_size', AJ_SUBSET_SIZE)

        params_diff_list = ['units_to_extract']

        pt_params_cnf = cnf.get('pt_params', {})
        for key in PT_PARAMS:
            if key not in params_diff_list and key in pt_params_cnf:
                PT_PARAMS[key] = pt_params_cnf[key]

        if 'units_to_extract' in pt_params_cnf:
            tmp = []
            for unit in pt_params_cnf['units_to_extract']:
                tmp.append(globals()[unit])
            PT_PARAMS['units_to_extract'] = tmp

    if METHOD != 'RMT':
        OVERRIDE = False if DO_REMAINING else OVERRIDE
    else:  # METHOD == 'RMT'
        GOLDEN_ROWS = False

    OUTPUT_DIR = OUTPUT_PATH + f"{METHOD}_{DATASET}_{'GL' if GOLDEN_ROWS else 'RM'}/"
    OUTPUT_FILE = OUTPUT_DIR + '_res.csv'

    NUM_PROCESSORS = multiprocessing.cpu_count() // 2 if NUM_PROCESSORS == 0 else NUM_PROCESSORS

    if OVERRIDE:
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    if METHOD == 'PT':
        get_pattern('run_pattern', PT_PARAMS, verbose=False)
    elif METHOD == 'AJ':
        get_pattern('run_aj', {'subset_size': AJ_SUBSET_SIZE, 'num_subsets': AJ_NUM_SUBSET})
    elif METHOD == 'RMT':
        row_matching_test()
    else:
        raise NotImplementedError()
