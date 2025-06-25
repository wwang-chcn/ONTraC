from optparse import Values
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from ..GNN import evaluate, predict, save_graph_pooling_results, set_seed, train
from ..log import *
from ..model import GNN
from ..niche_net import construct_niche_network, ct_coding_adjust, gen_samples_yaml
from ..niche_trajectory import NTScore_table, get_niche_NTScore, niche_to_cell_NTScore
from ..optparser import *
from ..preprocessing.pp_control import preprocessing_gnn, preprocessing_nn, preprocessing_nt
from ..train import GNNBatchTrain
from ..train.inspect_funcs import get_inspect_funcs
from ..utils import get_rel_params, read_yaml_file


def load_parameters(opt_validate_func: Callable, prepare_optparser_func: Callable) -> Values:
    """
    Load parameters.
    :param opt_validate_func: validate function.
    :param prepare_optparser_func: prepare optparser function.
    :return: options.
    """
    options = opt_validate_func(prepare_optparser_func())

    return options


def niche_network_construct(options: Values) -> None:
    """
    Niche network construct process.
    :param options: options.
    :return: None.
    """

    info('------------- Niche network construct --------------- ')

    # load input information
    meta_data_df, embedding_df, ct_coding_df = preprocessing_nn(
        meta_input=options.meta_input,
        NN_dir=options.NN_dir,
        exp_input=options.exp_input,
        embedding_input=options.embedding_input,
        low_res_exp_input=options.low_res_exp_input,
        deconvoluted_ct_composition=options.deconvoluted_ct_composition,
        deconvoluted_exp_input=options.deconvoluted_exp_input,
        resolution=options.resolution,
        dc_method=options.dc_method,
        dc_ct_num=options.dc_ct_num,
        gen_ct_embedding=options.embedding_adjust,
    )

    id_name = meta_data_df.columns[0]

    # construct niche network
    construct_niche_network(meta_data_df=meta_data_df,
                            ct_coding_df=ct_coding_df,
                            save_dir=options.NN_dir,
                            n_neighbors=options.n_neighbors,
                            n_local=options.n_local)

    # cell type coding adjust
    deconvoluted_exp_input = options.deconvoluted_exp_input if options.deconvoluted_exp_input is not None else Path(
        options.NN_dir).joinpath('celltype_x_gene_deconvolution.csv')
    deconvoluted_exp_input = deconvoluted_exp_input if id_name == 'Spot_ID' else None
    if options.embedding_adjust:
        ct_coding_adjust(NN_dir=options.NN_dir,
                         meta_data_df=meta_data_df,
                         embedding_df=embedding_df,
                         deconvoluted_exp_input=deconvoluted_exp_input,
                         sigma=options.sigma)

    # generate samples.yaml to indicate file paths for each sample
    gen_samples_yaml(meta_data_df=meta_data_df, save_dir=options.NN_dir)
    info('------------ Niche network construct end ------------ ')


def gnn(options: Values) -> None:
    """
    GNN training and prediction process.
    :param options: options.
    :return: None.
    """

    info('------------------------ GNN ------------------------ ')
    # load data
    dataset, sample_loader, meta_data_df = preprocessing_gnn(NN_dir=options.NN_dir, batch_size=options.batch_size)
    # random seed
    set_seed(seed=options.seed)
    # device
    device = options.device
    # build model
    input_feats = dataset.num_features
    model = GNN(input_feats=input_feats,
                hidden_feats=options.hidden_feats,
                k=options.k,
                n_gcn_layers=options.n_gcn_layers,
                exponent=options.beta)
    # train
    loss_weight_args: Dict[str, float] = {
        key: value
        for key, value in options.__dict__.items() if key.endswith('loss_weight')
    }
    batch_train = train(nn_model=model,
                        BatchTrain=GNNBatchTrain,
                        sample_loader=sample_loader,
                        device=device,
                        max_epochs=options.epochs,
                        max_patience=options.patience,
                        min_delta=options.min_delta,
                        min_epochs=options.min_epochs,
                        lr=options.lr,
                        save_dir=options.GNN_dir,
                        inspect_funcs=get_inspect_funcs(),
                        **loss_weight_args)
    # evaluate
    evaluate(batch_train=batch_train)
    # predict
    consolidate_z_array, consolidate_s_array, _ = predict(output_dir=options.GNN_dir,
                                                          batch_train=batch_train,
                                                          dataset=dataset)
    # save results
    if consolidate_z_array is not None and consolidate_s_array is not None:
        save_graph_pooling_results(meta_data_df=meta_data_df,
                                   dataset=dataset,
                                   rel_params=get_rel_params(NN_dir=options.NN_dir,
                                                             params=read_yaml_file(f'{options.NN_dir}/samples.yaml')),
                                   consolidate_z_array=consolidate_z_array,
                                   consolidate_s_array=consolidate_s_array,
                                   output_dir=options.GNN_dir)
    info('--------------------- GNN end ---------------------- ')


def niche_trajectory_construct(options: Values) -> None:
    """
    Pseudotime calculateion process.
    :param options: options.
    :return: None.
    """

    info('----------------- Niche trajectory ------------------ ')
    # load data
    meta_data_df, niche_level_niche_cluster_assign_df, consolidate_out_adj_array = preprocessing_nt(
        NN_dir=options.NN_dir, GNN_dir=options.GNN_dir)

    params = read_yaml_file(f'{options.NN_dir}/samples.yaml')
    rel_params = get_rel_params(NN_dir=options.NN_dir, params=params)

    niche_cluster_score, niche_level_NTScore_df = get_niche_NTScore(
        trajectory_construct_method=options.trajectory_construct,
        niche_level_niche_cluster_assign_df=niche_level_niche_cluster_assign_df,
        niche_adj_matrix=consolidate_out_adj_array)
    np.savetxt(fname=f'{options.NT_dir}/niche_cluster_score.csv.gz', X=niche_cluster_score, delimiter=',')

    cell_level_NTScore_df = niche_to_cell_NTScore(meta_data_df=meta_data_df,
                                                  niche_level_NTScore_df=niche_level_NTScore_df,
                                                  rel_params=rel_params)

    NTScore_table(save_dir=options.NT_dir,
                  meta_data_df=meta_data_df,
                  niche_level_NTScore_df=niche_level_NTScore_df,
                  cell_level_NTScore_df=cell_level_NTScore_df,
                  rel_params=rel_params)
    info('--------------- Niche trajectory end ---------------- ')
