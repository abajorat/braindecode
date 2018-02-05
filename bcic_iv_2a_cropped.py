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
log = logging.getLogger(__name__)


def preprocessing(data_folder, subject_id, low_cut_hz):
    global train_set, test_set, valid_set, n_classes, n_chans
    global n_iters, input_time_length
    n_iters = 5000
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

    stop_criterion = Or([MaxEpochs(20),
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
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    print(exp.rememberer)
    # return exp, {"cost": exp.rememberer.lowest_val}
    return exp.rememberer.lowest_val


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss = train(cfg)

        return ({
            'loss': loss}
        )
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = '/home/andresp/braindecode/data/'
    subject_id = 1 # 1-9
    low_cut_hz = 4 # 0 or 4
    cuda = True
    preprocessing(data_folder, subject_id, low_cut_hz)
    cs = cs.create_config_space()
    if True:
        scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": n_iters,
                         "cs": cs,
                         "deterministic": "true",
                         "output_dir": ""})

        smac = SMAC(scenario=scenario, tae_runner=train)
        smac.optimize()
    else:
        nameserver, ns = utils.start_local_nameserver()

        # starting the worker in a separate thread
        w = WorkerWrapper(nameserver=nameserver, ns_port=ns)
        w.run(background=True)

        CG = RandomSampling(cs)

        # instantiating Hyperband with some minimal configuration
        HB = HpBandSter(
            config_generator=CG,
            run_id='0',
            eta=2,  # defines downsampling rate
            min_budget=1,  # minimum number of epochs / minimum budget
            max_budget=127,  # maximum number of epochs / maximum budget
            nameserver=nameserver,
            ns_port=ns,
            job_queue_sizes=(0, 1),
        )
        # runs one iteration if at least one worker is available
        res = HB.run(1, min_n_workers=1)

        # shutdown the worker and the dispatcher
        HB.shutdown(shutdown_workers=True)

        # extract incumbent trajectory and all evaluated learning curves
        traj = res.get_incumbent_trajectory()
        wall_clock_time = []
        cum_time = 0
    # exp = train(cs)
    # log.info("Last 10 epochs")
    # log.info("\n" + str(exp.epochs_df.iloc[-10:]))
