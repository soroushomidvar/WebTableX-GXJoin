import time

from Transformation.ExtendedPattern import ExtendedPattern
from pattern import Placeholders
from pattern import Unifier
from pattern import Extracter
from Transformation.Blocks.LiteralPatternBlock import LiteralPatternBlock
from Transformation.Blocks.PositionPatternBlock import PositionPatternBlock
from Transformation.Blocks.TokenPatternBlock import TokenPatternBlock
from Transformation.Blocks.SplitSubstrPatternBlock import SplitSubstrPatternBlock
from Transformation.Blocks.TwoCharSplitSubstrPatternBlock import TwoCharSplitSubstrPatternBlock

# PARAM LIST WILL BE OBTAINED FROM HERE, MUST BE UPDATED. But values are Used only when value are not passed from main
DEF_PARAMS = {
    'max_tokens': 3,  # maximum number of allowed placeholders
    'max_blocks': 3,  # maximum number of allowed blocks (either placeholder or literal)
    'generalize': False,
    'units_to_extract': [LiteralPatternBlock, PositionPatternBlock, TokenPatternBlock, SplitSubstrPatternBlock],  # literal must be included
    'token_splitters': [' ', ],  # set to None to disable. Break placeholders into other placeholders based on these chars
    'remove_duplicate_patterns': True,  # After generating all possible transformation, delete the duplicates, may take time
    'switch_literals_placeholders': False,  # Replace placeholder with literals and add them as new pattern
    'only_first_match': False,  # Take only first match for the placeholder or look for all of possible matches.
    'prioritize_equal_coverage_patterns': False,  # When generating the covering set, in case of equal coverage, consider other factors
    'extended_transformations': False,  # Applying extending functions by Sorush
    'extension_mode': 'None',
    'extension_index': 1,
}


def get_patterns(rows, params={}, table_name='', extras={}, verbose=True):

    for key in DEF_PARAMS:
        if key not in params:
            params[key] = DEF_PARAMS[key]

    # rows = {
    #     'arash dargahi nobari': ['arash d. nobari'],
    #     # 'arash dargahi': ['a. dargahi'],
    #     # 'arash dargahi nobar': ['arash d. nobar'],
    # }

    sum_inp = 0
    sum_outs = 0
    cnt_in_out = 0
    for inp, outs in rows.items():
        for out in outs:
            sum_inp += len(inp)
            sum_outs += len(out)
            cnt_in_out += 1

    start_time = time.time()

    placeholder_stats = {'cnt_all_generated_placeholder_comb': 0,'cnt_all_remaining_placeholder_comb': 0,'cnt_all_removed_placeholder_comb': 0,'cnt_init_remaining_placeholder_comb': 0,'cnt_splitted_remaining_placeholder_comb': 0,'cnt_removed_init_placeholder_comb': 0,'cnt_removed_splitted_placeholder_comb': 0,}

    # The extracted patterns are basic patterns, only indicating the place holders
    all_basic_patterns = Placeholders.get_placeholders(rows, params['max_tokens'], params['token_splitters'], verbose,
                                                       {
                                                            'break_each_block_separate': False,
                                                            'add_literals': params['switch_literals_placeholders'],
                                                            'only_first_match': params['only_first_match'],
                                                            'max_blocks': params['max_blocks'],
                                                        },
                                                       placeholder_stats)

    placeholder_gen_time_end = time.time()




    # Get all patterns
    all_patterns_list = Extracter.extract_patterns(rows, all_basic_patterns, params['units_to_extract'], verbose)

    cnt_placeholder_comb = len(all_basic_patterns)
    cnt_all_patterns = len(all_patterns_list)
    cnt_dup_patterns_removed = 0

    extract_pat_time_end = time.time()

    if params['remove_duplicate_patterns']:
        all_patterns = set(all_patterns_list)
        cnt_dup_patterns_removed = cnt_all_patterns - len(all_patterns)
    else:
        all_patterns = all_patterns_list

    print(f"{table_name} --> {cnt_placeholder_comb:,} Place holder comb. and {cnt_all_patterns:,} total transformations. {cnt_dup_patterns_removed:,} duplicate removed, {len(all_patterns):,} remaining.")
    duplicate_pat_remove_time_end = time.time()

    inputs, pat_inp, inp_pat, patterns, gen_time, effective_gens, pattern_counts =\
        Unifier.generate_similar_patterns(all_patterns, rows, params['generalize'], verbose)
    pat_applying_time_end = time.time()

    # Get covering set
    final_res = Unifier.get_covering_set(inputs, pat_inp, patterns, verbose, params['prioritize_equal_coverage_patterns'])

    end_time = time.time()

    extend_res = None

    if params['extended_transformations']:
        from pattern import Extender
        extend_res = []

        # pat_inp, inp_pat, patterns = [], [], []

        # print(f"Getting Extended Patterns - {table_name}")
        for r in final_res:
            tr = r[2]
            extended_list = Extender.generalizer(tr, extras['extension_index'], extras['file_path'],
                                                 extras['src'], extras['target'], extras['extension_mode'])
            inputs2, pat_inp2, inp_pat2, patterns2, gen_time2, effective_gens2, pattern_counts2 = \
                Unifier.generate_similar_patterns(extended_list, rows, False, verbose)
            new_cov = set()
            for p in pat_inp2:
                new_cov |= p

            assert len(new_cov) >= r[1]



            # new_pat = f"Gen {tr}"
            new_pat = ExtendedPattern(list(extended_list), tr)
            idx = len(patterns)
            patterns.append(new_pat)


            assert patterns[idx] == new_pat
            pat_inp.append(new_cov)
            assert len(patterns) == len(pat_inp)

            for i in range(len(inp_pat)):
                if i in new_cov:
                    inp_pat[i].add(idx)


            extend_res.append({
                'transformation': new_pat,
                'old_tr': tr,
                'extended_list': extended_list,
                'each_cover': [{patterns2[i]: [inputs[c] for c in cov]} for i, cov in enumerate(pat_inp2)],
                'old_pat_inp': r[3],
                'new_pat_inp': [inputs2[i] for i in new_cov],
                'old_cov': r[1],
                'new_cov': len(new_cov),
            })




        # print( f"Applying extended patterns - {table_name} --> {len(extended_list)} total transformations.")




        final_res = Unifier.get_covering_set(inputs, pat_inp, patterns, verbose, params['prioritize_equal_coverage_patterns'])

    extending_end_time = time.time()

    ####

    # print transformations
    if verbose:
        for res in final_res:
            print(f"id: {res[0]}, covered:{res[1]}/{len(inputs)}, pattern: {res[2]}")
            # for ip in pat_inp[res[0]]:
            #     print(f"   |-> '{inputs[ip][0]}' -> '{inputs[ip][1]}'")



    return {
        'patterns': final_res,
        'ranked': sorted(final_res, reverse=True, key=lambda item: item[1]),
        'input_len': len(inputs),
        'covered': sum(1 for p in inp_pat if len(p) > 0),
        'runtime': extending_end_time - start_time,
        'effective_gens': effective_gens,
        'gen_time': gen_time,
        'max_tokens': params['max_tokens'],

        'avg_len_input': 0 if cnt_in_out == 0 else sum_inp/cnt_in_out,
        'avg_len_output': 0 if cnt_in_out == 0 else sum_outs/cnt_in_out,

        'cnt_placeholder_comb': cnt_placeholder_comb,
        'cnt_all_generated_placeholder_comb': placeholder_stats['cnt_all_generated_placeholder_comb'],
        'cnt_all_remaining_placeholder_comb': placeholder_stats['cnt_all_remaining_placeholder_comb'],
        'cnt_all_removed_placeholder_comb': placeholder_stats['cnt_all_removed_placeholder_comb'],
        'cnt_init_remaining_placeholder_comb': placeholder_stats['cnt_init_remaining_placeholder_comb'],
        'cnt_splitted_remaining_placeholder_comb': placeholder_stats['cnt_splitted_remaining_placeholder_comb'],
        'cnt_removed_init_placeholder_comb': placeholder_stats['cnt_removed_init_placeholder_comb'],
        'cnt_removed_splitted_placeholder_comb': placeholder_stats['cnt_removed_splitted_placeholder_comb'],

        'placeholder_gen_time': placeholder_gen_time_end - start_time,
        'extract_pat_time': extract_pat_time_end - placeholder_gen_time_end,
        'duplicate_pat_remove_time': duplicate_pat_remove_time_end - extract_pat_time_end,
        'pat_applying_time': pat_applying_time_end - duplicate_pat_remove_time_end,
        'get_covering_set_time': end_time - pat_applying_time_end,
        'extending_time': extending_end_time - end_time,

        'cnt_all_patterns': cnt_all_patterns,
        'cnt_dup_patterns_removed': cnt_dup_patterns_removed,
        'cnt_pattern_remaining': len(all_patterns),

        'cnt_all_patterns_all_rows': pattern_counts[0],
        'cnt_patterns_successful': pattern_counts[1],
        'cnt_patterns_failed': pattern_counts[0] - pattern_counts[1],
        'cnt_patterns_hit_cache': pattern_counts[2],

        'params': params,

        'extend_res': extend_res,
    }

