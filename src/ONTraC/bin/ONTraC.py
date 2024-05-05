#!/usr/bin/env python

import random
import sys
from typing import Optional

import numpy as np
import torch

from ..log import *
from ..model import GraphPooling
from ..optparser import opt_ontrac_validate, prepare_ontrac_optparser
from ..run.processes import *
from ..train import GPBatchTrain, SubBatchTrainProtocol
from ..train.inspect_funcs import loss_record
from ..utils import device_validate, write_version_info
from ..utils.niche_net_constr import construct_niche_network, gen_samples_yaml, load_original_data


# ------------------------------------
# Functions
# ------------------------------------
def get_inspect_funcs() -> Optional[list[Callable]]:
    """
    Inspect function list
    :param output_dir: output dir
    :param epoch_filter: epoch filter
    :return: list of inspect functions
    """
    return [loss_record]


# ------------------------------------
# Main Function
# ------------------------------------
def main() -> None:
    """
    main function
    Input data files information should be stored in a YAML file.
    """

    # write version information
    write_version_info()
    
    # prepare options
    options = load_parameters(opt_validate_func=opt_ontrac_validate, prepare_optparser_func=prepare_ontrac_optparser)

    # ----- Niche Network Construct -----
    # load original data
    ori_data_df = load_original_data(options=options)

    # define edges for each sample
    construct_niche_network(options=options, ori_data_df=ori_data_df)

    # save samples.yaml
    gen_samples_yaml(options=options, ori_data_df=ori_data_df)

    # ----- Graph Pooling -----
    # device
    device: torch.device = device_validate(device_name=options.device)
    # load data
    dataset, sample_loader = load_data(options=options)
    # random seed
    n_seed = t_seed = r_seed = options.seed
    random.seed(a=r_seed)
    torch.manual_seed(seed=t_seed)
    np.random.seed(seed=n_seed)
    # train
    inspect_funcs_list = get_inspect_funcs()
    batch_train: SubBatchTrainProtocol = train(nn_model=GraphPooling,
                                               options=options,
                                               BatchTrain=GPBatchTrain,
                                               device=device,
                                               dataset=dataset,
                                               sample_loader=sample_loader,
                                               inspect_funcs=inspect_funcs_list,
                                               model_name='GraphPooling')
    # evaluate
    evaluate(batch_train=batch_train, model_name='GraphPooling')
    # predict
    consolidate_s_array, consolidate_out_adj_array = predict(output_dir=options.GNN_dir,
                                                             batch_train=batch_train,
                                                             dataset=dataset,
                                                             model_name='GraphPooling')
    # niche cluster
    if consolidate_s_array is not None:
        graph_pooling_output(ori_data_df=ori_data_df,
                             dataset=dataset,
                             rel_params=get_rel_params(
                                 options=options, params=read_yaml_file(f'{options.preprocessing_dir}/samples.yaml')),
                             consolidate_s_array=consolidate_s_array,
                             output_dir=options.GNN_dir)

    # ----- NT score -----
    if consolidate_s_array is not None and consolidate_out_adj_array is not None:
        NTScore(options=options,
                dataset=dataset,
                consolidate_s_array=consolidate_s_array,
                consolidate_out_adj_array=consolidate_out_adj_array)


# ------------------------------------
# Program running
# ------------------------------------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupts me! ;-) See you ^.^!\n")
        sys.exit(0)
