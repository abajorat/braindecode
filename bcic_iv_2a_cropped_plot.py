import logging
import os.path
# import time
from collections import OrderedDict
import sys

# import numpy as np
import torch.nn.functional as F
from torch import optim
import torch as th
import cs
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
# from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from hpbandster.distributed import utils
from hpbandster.config_generators import RandomSampling
from hpbandster.HB_master import HpBandSter
from hpbandster.distributed.worker import Worker
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
import ConfigSpace as CS
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario


import matplotlib.pyplot as plt
log = logging.getLogger(__name__)


def preprocessing(data_folder, subject_id, low_cut_hz):
    global train_set, test_set, valid_set, n_classes, n_chans
    global n_iters, input_time_length
# def run_exp(data_folder, subject_id, low_cut_hz, model, cuda):
    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')

    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()
    train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                         'EOG-central', 'EOG-right'])
    assert len(train_cnt.ch_names) == 22
    # lets convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, 38, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                       'EOG-central', 'EOG-right'])
    assert len(test_cnt.ch_names) == 22
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, 38, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    ival = [-500, 4000]
    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)
    set_random_seeds(seed=20190706, cuda=cuda)
    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    input_time_length=1000


def train(config):
    cuda = True
    model = config['model']

    # if model == 'shallow':
        # model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            # final_conv_length=30).create_network()
    if model == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=2, config=config).create_network()

    to_dense_prediction_model(model)
    if cuda:
        model.cuda()

    log.info("Model: \n{:s}".format(str(model)))
    dummy_input = np_to_var(train_set.X[:1, :, :, None])
    if cuda:
        dummy_input = dummy_input.cuda()
    out = model(dummy_input)

    n_preds_per_input = out.cpu().data.numpy().shape[2]

    optimizer = optim.Adam(model.parameters())

    iterator = CropsFromTrialsIterator(batch_size=60,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    stop_criterion = Or([MaxEpochs(800),
                         NoDecrease('valid_misclass', 80)])

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=False, cuda=cuda)
    exp.run()
    print(exp.rememberer)
    # return exp, {"cost": exp.rememberer.lowest_val}
    return exp.rememberer.lowest_val


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = '/home/andresp/braindecode/data/'
    subject_id = 1 # 1-9
    low_cut_hz = 4 # 0 or 4
    cuda = True
    preprocessing(data_folder, subject_id, low_cut_hz)
    default_config = {
                 "model":'deep',
                 "n_filters_time":25,
                 "n_filters_spat":25,
                 "filter_time_length":10,
                 "pool_time_length":3,
                 "pool_time_stride":3,
                 "n_filters_2":50,
                 "filter_length_2":10,
                 "n_filters_3":100,
                 "filter_length_3":10,
                 "n_filters_4":200,
                 "filter_length_4":10,
                 "first_nonlin":"elu",
                 "first_pool_mode":'max',
                 "first_pool_nonlin":"identity",
                 "later_nonlin":"elu",
                 "later_pool_mode":'max',
                 "later_pool_nonlin":"identity",
                 "drop_prob":0.5,
                 "double_time_convs":False,
                 "split_first_layer":True,
                 "batch_norm":True,
                 "batch_norm_alpha":0.1,
                 "stride_before_pool":False}
    train(default_config)
    best_config = {
                 "model":'deep',
                 "n_filters_time":55,
                 "n_filters_spat":25,
                 "filter_time_length":10,
                 "pool_time_length":8,
                 "pool_time_stride":2,
                 "n_filters_2":19,
                 "filter_length_2":8,
                 "n_filters_3":265,
                 "filter_length_3":5,
                 "n_filters_4":19,
                 "filter_length_4":12,
                 "first_nonlin":"elu",
                 "first_pool_mode":'mean',
                 "first_pool_nonlin":"identity",
                 "later_nonlin":"tanh",
                 "later_pool_mode":'mean',
                 "later_pool_nonlin":"identity",
                 "drop_prob":0.1630354584938897,
                 "double_time_convs":False,
                 "split_first_layer":True,
                 "batch_norm":True,
                 "batch_norm_alpha":0.6891877533370003,
                 "stride_before_pool":False}

    train(best_config)
    x_axis = []
    y_axis = []
    previous_line = "\n"
    with open("validation_trajectory", "r") as f:
        for line in f:
            if line != "\n":
                if previous_line == "\n":
                    x_axis.append(line.strip().split("\t"))
                else:
                    y_axis.append(line.strip().split("\t"))
            previous_line = line
        f.close()
    x_axis = [list(map(float, sublist)) for sublist in x_axis]
    y_axis = [list(map(float, sublist)) for sublist in y_axis]
    print(x_axis)
    print(y_axis)
    plt.plot(x_axis[0], y_axis[0], label="default")
    plt.plot(x_axis[1], y_axis[1], label="improvement")

    plt.xlabel('Epochs')
    plt.ylabel('Validation misclassification')
    plt.title('Performance comparison')
    plt.legend()
    plt.show()
