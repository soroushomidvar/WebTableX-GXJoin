import math
import random
import pathlib
import os
from data_processor import DataLoader as dl
from data_processor import Matcher as matcher
BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()

runs = 10
sample_sizes = [4] #,10, 20, 40]
datasets= ["BM", "FF"]
DS_PATH = ""
OUT_PATH = ""
SAMPLING_RATE = 0.1

for run in range(runs):
    for dataset in datasets:
        for sample_size in sample_sizes:
            # set dataset path
            if dataset=="BM":
                DS_PATH = str(BASE_PATH / 'data/autojoin-Benchmark/')  
            else:
                DS_PATH = str(BASE_PATH / 'data/FlashFill/') 
            
            # set output path
            #out_rel_path =  'data/_samples/' + dataset + '/' + str(sample_size) + '/'
            out_rel_path =  'data/_samples/' + dataset + '/' + "run " + str(run) + '/' + str(sample_size) + '/'
            OUT_PATH = str(BASE_PATH / out_rel_path) 
            #OUT_PATH = os.path.join(BASE_PATH, out_rel_path)

            # set sample rate
            SAMPLING_RATE = sample_size/100
            
            print("DS_PATH: " + DS_PATH)
            print("OUT_PATH: " + OUT_PATH)

            tables, all_tables = dl.get_tables_from_dir(DS_PATH, [], make_lower=False)
            res = matcher.get_matching_tables_golden(tables, bidi=False)

            for item in res['items']:
                tbl_name = item['src_table'][4:]
                print(f"Sampling {tbl_name} with ratio of {SAMPLING_RATE}")
                rows, is_swapped = matcher.get_matching_rows_golden(tables, item, swap_src_target=False, force_swap=0)
                assert not is_swapped

                #size = math.ceil(SAMPLING_RATE * len(rows))
                size = max(math.ceil(SAMPLING_RATE * len(rows)), 2)
                keys = random.sample(list(rows), size)
                sample_rows = {}
                FILE_OUT_PATH = OUT_PATH + "/" +tbl_name+".csv"
                print("FILE_OUT_PATH: " + FILE_OUT_PATH)

                # Create the directory if it doesn't exist
                if not os.path.exists(OUT_PATH):
                    os.makedirs(OUT_PATH)

                with open(FILE_OUT_PATH, 'w') as f:
                    print("source,target", file=f)
                    for k in keys:
                        val = rows[k].pop()
                        assert "," not in k and "," not in val
                        print(f"{k},{val}", file=f)

