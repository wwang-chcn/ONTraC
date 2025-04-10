from ONTraC.constants import IO_OPTIONS
from ONTraC.optparser._IO import IOOptionsCollection, io_dicts_to_io_options_collection


def test_ontrac_ioc() -> None:
    """
    Test the IO options collection for ONTraC.
    """

    ioc = io_dicts_to_io_options_collection(IO_OPTIONS['ONTraC'])
    assert isinstance(ioc, IOOptionsCollection)
    assert ioc.has_io_option('input')
    assert ioc.get_io_option_attr('input') == 'required'
    assert ioc.has_io_option('NN_dir')
    assert ioc.get_io_option_attr('NN_dir') == 'overwrite'
    assert ioc.has_io_option('GNN_dir')
    assert ioc.get_io_option_attr('GNN_dir') == 'overwrite'
    assert ioc.has_io_option('NT_dir')
    assert ioc.get_io_option_attr('NT_dir') == 'overwrite'
    # wrong options
    assert not ioc.has_io_option('output')
    assert not ioc.has_io_option('log')


def test_ontrac_nn_ioc() -> None:
    """
    Test the IO options collection for ONTraC_NN.
    """

    ioc = io_dicts_to_io_options_collection(IO_OPTIONS['ONTraC_NN'])
    assert isinstance(ioc, IOOptionsCollection)
    assert ioc.has_io_option('input')
    assert ioc.get_io_option_attr('input') == 'required'
    assert ioc.has_io_option('NN_dir')
    assert ioc.get_io_option_attr('NN_dir') == 'overwrite'
    # wrong options
    assert not ioc.has_io_option('GNN_dir')
    assert not ioc.has_io_option('NT_dir')
    assert not ioc.has_io_option('output')
    assert not ioc.has_io_option('log')


def test_ontrac_gnn_ioc() -> None:
    """
    Test the IO options collection for ONTraC_GNN.
    """

    ioc = io_dicts_to_io_options_collection(IO_OPTIONS['ONTraC_GNN'])
    assert isinstance(ioc, IOOptionsCollection)
    assert ioc.has_io_option('NN_dir')
    assert ioc.get_io_option_attr('NN_dir') == 'required'
    assert ioc.has_io_option('GNN_dir')
    assert ioc.get_io_option_attr('GNN_dir') == 'overwrite'
    # wrong options
    assert not ioc.has_io_option('input')
    assert not ioc.has_io_option('NT_dir')
    assert not ioc.has_io_option('output')
    assert not ioc.has_io_option('log')


def test_ontrac_nt_ioc() -> None:
    """
    Test the IO options collection for ONTraC_NT.
    """

    ioc = io_dicts_to_io_options_collection(IO_OPTIONS['ONTraC_NT'])
    assert isinstance(ioc, IOOptionsCollection)
    assert ioc.has_io_option('NN_dir')
    assert ioc.get_io_option_attr('NN_dir') == 'required'
    assert ioc.has_io_option('GNN_dir')
    assert ioc.get_io_option_attr('GNN_dir') == 'required'
    assert ioc.has_io_option('NT_dir')
    assert ioc.get_io_option_attr('NT_dir') == 'overwrite'
    # wrong options
    assert not ioc.has_io_option('input')
    assert not ioc.has_io_option('output')
    assert not ioc.has_io_option('log')


def test_ontrac_gt_ioc() -> None:
    """
    Test the IO options collection for ONTraC_GT.
    """

    ioc = io_dicts_to_io_options_collection(IO_OPTIONS['ONTraC_GT'])
    assert isinstance(ioc, IOOptionsCollection)
    assert ioc.has_io_option('NN_dir')
    assert ioc.get_io_option_attr('NN_dir') == 'required'
    assert ioc.has_io_option('GNN_dir')
    assert ioc.get_io_option_attr('GNN_dir') == 'overwrite'
    assert ioc.has_io_option('NT_dir')
    assert ioc.get_io_option_attr('NT_dir') == 'overwrite'
    # wrong options
    assert not ioc.has_io_option('input')
    assert not ioc.has_io_option('output')
    assert not ioc.has_io_option('log')


def test_ontrac_analysis_ioc() -> None:
    """
    Test the IO options collection for ONTraC_analysis.
    """

    ioc = io_dicts_to_io_options_collection(IO_OPTIONS['ONTraC_analysis'])
    assert isinstance(ioc, IOOptionsCollection)
    assert ioc.has_io_option('NN_dir')
    assert ioc.get_io_option_attr('NN_dir') == 'required'
    assert ioc.has_io_option('GNN_dir')
    assert ioc.get_io_option_attr('GNN_dir') == 'optional'
    assert ioc.has_io_option('NT_dir')
    assert ioc.get_io_option_attr('NT_dir') == 'optional'
    assert ioc.has_io_option('output')
    assert ioc.get_io_option_attr('output') == 'optional-overwrite'
    assert ioc.has_io_option('log')
    assert ioc.get_io_option_attr('log') == 'optional'
    # wrong options
    assert not ioc.has_io_option('input')


def test_io_options():

    test_ontrac_ioc()
    test_ontrac_nn_ioc()
    test_ontrac_gnn_ioc()
    test_ontrac_nt_ioc()
    test_ontrac_gt_ioc()
    test_ontrac_analysis_ioc()
