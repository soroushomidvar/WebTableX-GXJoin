import copy
import multiprocessing
import os
import pathlib
import sys
from io import StringIO

from data_processor import DataLoader as dl
from pattern import joiner
from evaluator.JoinEval import JoinEval
from data_processor import Matcher as matcher
from pattern import Finder
from Transformation.Blocks.LiteralPatternBlock import LiteralPatternBlock
from Transformation.Blocks.PositionPatternBlock import PositionPatternBlock
from Transformation.Blocks.TokenPatternBlock import TokenPatternBlock
from Transformation.Blocks.SplitSubstrPatternBlock import SplitSubstrPatternBlock

params = {
    'max_tokens': 3,  # maximum number of allowed placeholders
    'max_blocks': 3,  # maximum number of allowed blocks (either placeholder or literal)
    'generalize': False,
    'sample_size': None,
    # 'sample_size': ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', 'kmeans'],

    'token_splitters': [' ', ],
    # set to None to disable. Break placeholders into other placeholders based on these chars
    'remove_duplicate_patterns': True,
    # After generating all possible transformation, delete the duplicates, may take time
    'switch_literals_placeholders': True,  # Replace placeholder with literals and add them as new pattern
    'only_first_match': False,  # Take only first match for the placeholder or look for all of possible matches.

    'units_to_extract': [LiteralPatternBlock, PositionPatternBlock, TokenPatternBlock, SplitSubstrPatternBlock],

    'prioritize_equal_coverage_patterns': False,

    'extended_transformations': False,

    'extension_mode': 'duplicating',
    'extension_index': 2,

    'src_tgt_swapping_sample_percentage': None,
    'src_tgt_swapping_sample_min_size': 5,

}

BASE_PATH = str(pathlib.Path(__file__).absolute().parent.parent.absolute())


GOLDEN_ROWS = True
SWAP_SRC_TARGET = True
ROW_MATCHING_N_START=4
ROW_MATCHING_N_END = 20
MULTI_CORE=True
NUM_PROCESSORS=multiprocessing.cpu_count()//2

MIN_SUPPORT=0

'''
An example of end2end join
'''

sum_p = multiprocessing.Value('d', 0)
sum_r = multiprocessing.Value('d', 0)
sum_f = multiprocessing.Value('d', 0)
sum_cnt = multiprocessing.Value('i', 0)


def get_tables_from_dir(ds_path, tbl_names, make_lower=False, verbose=False, prefix="smpl-"):
    tables = {}
    all_tables = []
    assert os.path.isdir(ds_path)
    dirs = [dI for dI in pathlib.Path(ds_path).glob(f"{prefix}*") if os.path.isdir(dI)]
    for ddd in dirs:
        dir = ddd.name
        if len(tbl_names) > 0 and dir not in tbl_names:
            if verbose: print(f"*** {dir} not in specified names")
            continue
        if '_' in dir:
            raise Exception("_ cannot be in dir name: "+dir)
        if verbose: print("Reading "+dir)
        ds_dir = str(ds_path)+'/' + dir
        assert os.path.exists(ds_dir + "/source.csv")
        assert os.path.exists(ds_dir + "/target.csv")
        assert os.path.exists(ds_dir + "/rows.txt")

        has_gt = os.path.exists(ds_dir + "/ground truth.csv")

        res = {
            'src': {'name': 'src_'+dir, 'titles': None, 'items': []},
            'target': {'name': 'target_'+dir, 'titles': None, 'items': []},
            'name': dir
        }

        if has_gt:
            res['GT'] = {'titles': None, 'items': []}


        with open(ds_dir + "/source.csv") as f:
            res['src']['titles'] = f.readline().strip().split(',')
            if make_lower:
                res['src']['items'] = [line.lower().strip().split(',') for line in f.readlines()]
            else:
                res['src']['items'] = [line.strip().split(',') for line in f.readlines()]

        with open(ds_dir + "/target.csv") as f:
            res['target']['titles'] = f.readline().strip().split(',')
            if make_lower:
                res['target']['items'] = [line.lower().strip().split(',') for line in f.readlines()]
            else:
                res['target']['items'] = [line.strip().split(',') for line in f.readlines()]

        if has_gt:
            with open(ds_dir + "/ground truth.csv") as f:
                res['GT']['titles'] = f.readline().strip().split(',')
                if make_lower:
                    res['GT']['items'] = [line.lower().strip().split(',') for line in f.readlines()]
                else:
                    res['GT']['items'] = [line.strip().split(',') for line in f.readlines()]

        with open(ds_dir + "/rows.txt") as f:
            l = f.readline().strip().split(':')
            s = l[0]
            t = l[1]
            assert s in res['src']['titles']
            assert t in res['target']['titles']
            res['rows'] = {'src': s, 'target': t}

            l = f.readline().strip()
            assert l in ("source", "target")
            res['source_col'] = l

        if has_gt:
            assert len(res['GT']['titles']) == len(res['src']['titles']) + len(res['target']['titles'])

            change = not res['GT']['titles'][0].startswith("source-")

            for i in range(0, len(res['src']['titles'])):
                if change:
                    res['GT']['titles'][i] = 'source-' + res['GT']['titles'][i]

                assert res['GT']['titles'][i] == 'source-' + res['src']['titles'][i]

            for i in range(len(res['src']['titles']), len(res['GT']['titles'])):
                if change:
                    res['GT']['titles'][i] = 'target-' + res['GT']['titles'][i]

                assert res['GT']['titles'][i] == 'target-' + res['target']['titles'][i-len(res['src']['titles'])]

        tables[dir] = res
        all_tables.append(res['src'])
        all_tables.append(res['target'])
    return tables, all_tables


def run_join(ds_path, train_prefix, test_prefix):
    tables, all_tables = get_tables_from_dir(ds_path, [], make_lower=True, verbose=False, prefix=train_prefix)

    global sum_p
    global sum_r
    global sum_f
    global sum_cnt

    sum_p.value = 0
    sum_f.value = 0
    sum_r.value = 0
    sum_cnt.value = 0


    res = matcher.get_matching_tables_golden(tables, bidi=False)

    all_rows = []
    data = []
    all_has_gt = True

    for item in res['items']:

        has_GT = 'GT' in tables[item['src_table'][4:]]
        all_has_gt = all_has_gt and has_GT


        if GOLDEN_ROWS:
            if not has_GT:
                raise Exception(
                    'golden row matching cannot be used when no ground truth table exists for ' + item['src_table'][4:])

            rows, is_swapped = matcher.get_matching_rows_golden(tables, item, swap_src_target=SWAP_SRC_TARGET)
        else:
            rows, is_swapped = matcher.get_matching_rows_for_table(tables, item, ROW_MATCHING_N_START, ROW_MATCHING_N_END, swap_src_target=SWAP_SRC_TARGET)

        new_rows = {}
        for src, target in rows.items():
            new_rows[src] = [[t, t] for t in target]


        rr = {
            'col_info': item,
            'is_swapped': is_swapped,
            'rows': new_rows
        }
        all_rows.append(rr)

        rmu = None
        # if has_GT:
        #     rmu = RowMatcherUnit(tables, rr)
        #     print(rmu)

        params_new = copy.deepcopy(params)

        params_new['tbl_info'] = {
            "test_path": ds_path / f"{test_prefix}{item['src_table'][4+len(train_prefix):]}/",
            'is_swapped': is_swapped
        }


        if MULTI_CORE:
            data.append((item, rows, params_new, tables, None, rmu, False))
        else:
            run_pattern(item, rows, params_new, tables, None, rmu, verbose=False)


    if MULTI_CORE:

        # @TODO: Support spawn
        if sys.platform in ('win32', 'msys', 'cygwin'):
            print("fork based multi core processing works only on *NIX type operating systems.")
            sys.exit(1)

        from multiprocessing import get_context
        pool = get_context('fork').Pool(processes=NUM_PROCESSORS)
        # pool = multiprocessing.Pool(processes=NUM_PROCESSORS)

        rets = pool.starmap(run_pattern, data)
        pool.close()

        pool.join()



class NullIO(StringIO):
    def write(self, txt):
       pass

def run_pattern(item, rows, params, tables, table_name=None, rmu=None, verbose=None):
    if table_name is None:
        table_name = item['src_table'][4:]

    extras = {}
    if params['extended_transformations']:
        raise Exception("Not implemented")

    old_out = sys.stdout
    sys.stdout = NullIO()
    res = Finder.get_patterns(rows, params=params, table_name=table_name, extras=extras, verbose=verbose)
    sys.stdout = old_out


    trans = [t[2] for t in res['ranked']]

    src, target, gt = dl.basic_loader(params['tbl_info']['test_path'], make_lower=True)
    # Perform the join:
    joins = joiner.join(src, target, trans, params['tbl_info']['is_swapped'], min_support=MIN_SUPPORT)

    # Evaluate the join:
    je = JoinEval(joins, gt)
    # print(f"{table_name},{je.precision},{je.recall},{je.f1}")

    global sum_p
    global sum_r
    global sum_f
    global sum_cnt
    with sum_p.get_lock():
        sum_p.value += je.precision
    with sum_r.get_lock():
        sum_r.value += je.recall
    with sum_f.get_lock():
        sum_f.value += je.f1
    with sum_cnt.get_lock():
        sum_cnt.value += 1


def main():


    db_prefix = "AJ-Splitted--"
    ds_root = BASE_PATH + '/data/dtt/noisy/'

    for ds_path in sorted(pathlib.Path(ds_root).glob(f"{db_prefix}*")):
        # print(ds_path)
        assert os.path.isdir(ds_path)



        run_join(ds_path, "smpl-", "test-")
        c = sum_cnt.value
        # print("---------------")
        print(f"{ds_path.name},{sum_p.value/c},{sum_r.value/c},{sum_f.value/c}")



if __name__ == "__main__":
    main()



