import numpy as np
import pickle
import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import logging
from copy import deepcopy
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
import matplotlib.patches as mpatches

import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker
logging.basicConfig(level=logging.ERROR)


def create_config_space():
    cs = CS.ConfigurationSpace()

    stepDecay = CS.UniformIntegerHyperparameter("stepDecay_epochs_per_step",
                                                lower=1,
                                                default_value=16,
                                                upper=128,
                                                log=True)
    batch_size = CS.UniformIntegerHyperparameter("batch_size",
                                                 lower=8,
                                                 default_value=16,
                                                 upper=256,
                                                 log=True)

    num_layers = CS.UniformIntegerHyperparameter("num_layers",
                                                 lower=1,
                                                 default_value=2,
                                                 upper=4)
    num_units_0 = CS.UniformIntegerHyperparameter("num_units_0",
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256,
                                                  log=True)
    num_units_1 = CS.UniformIntegerHyperparameter("num_units_1",
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256,
                                                  log=True)
    num_units_2 = CS.UniformIntegerHyperparameter("num_units_2",
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256,
                                                  log=True)
    num_units_3 = CS.UniformIntegerHyperparameter("num_units_3",
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256,
                                                  log=True)
    adam_fraction = CS.UniformFloatHyperparameter("Adam_final_lr_fraction",
                                                  lower=1e-4,
                                                  default_value=1e-2,
                                                  upper=1,
                                                  log=True)
    adam_initial = CS.UniformFloatHyperparameter("Adam_initial_lr",
                                                 lower=1e-4,
                                                 default_value=1e-3,
                                                 upper=1e-2,
                                                 log=True)
    sgd_fraction = CS.UniformFloatHyperparameter("SGD_final_lr_fraction",
                                                 lower=1e-4,
                                                 default_value=1e-2,
                                                 upper=1,
                                                 log=True)
    sgd_initial = CS.UniformFloatHyperparameter("SGD_initial_lr",
                                                lower=1e-3,
                                                default_value=1e-1,
                                                upper=0.5,
                                                log=True)
    SGD_momentum = CS.UniformFloatHyperparameter("SGD_momentum",
                                                 lower=0.0,
                                                 default_value=0.9,
                                                 upper=0.99,
                                                 log=False)
    dropout_0 = CS.UniformFloatHyperparameter("dropout_0",
                                              lower=0.0,
                                              default_value=0.0,
                                              upper=0.5,
                                              log=False)
    dropout_1 = CS.UniformFloatHyperparameter("dropout_1",
                                              lower=0.0,
                                              default_value=0.0,
                                              upper=0.5,
                                              log=False)

    dropout_2 = CS.UniformFloatHyperparameter("dropout_2",
                                              lower=0.0,
                                              default_value=0.0,
                                              upper=0.5,
                                              log=False)
    dropout_3 = CS.UniformFloatHyperparameter("dropout_3",
                                              lower=0.0,
                                              default_value=0.0,
                                              upper=0.5,
                                              log=False)
    l2_reg_0 = CS.UniformFloatHyperparameter("l2_reg_0",
                                             lower=1e-6,
                                             default_value=1e-4,
                                             upper=1e-2,
                                             log=True)
    l2_reg_1 = CS.UniformFloatHyperparameter("l2_reg_1",
                                             lower=1e-6,
                                             default_value=1e-4,
                                             upper=1e-2,
                                             log=True)
    l2_reg_2 = CS.UniformFloatHyperparameter("l2_reg_2",
                                             lower=1e-6,
                                             default_value=1e-4,
                                             upper=1e-2,
                                             log=True)
    l2_reg_3 = CS.UniformFloatHyperparameter("l2_reg_3",
                                             lower=1e-6,
                                             default_value=1e-4,
                                             upper=1e-2,
                                             log=True)
    activation = CS.CategoricalHyperparameter("activation", ['relu', 'tanh'])
    optimizer = CS.CategoricalHyperparameter("optimizer", ['Adam', 'SGD'])
    learning_rate_schedule = CS.CategoricalHyperparameter("learning_" +
                                                          "rate_schedule",
                                                          ['ExponentialDecay',
                                                           'StepDecay'])
    loss_function = CS.CategoricalHyperparameter("loss_function",
                                                 ['categorical' +
                                                  '_crossentropy'])
    output_activation = CS.CategoricalHyperparameter("output_activation",
                                                     ['softmax'])
    cs.add_hyperparameter(batch_size)
    cs.add_hyperparameter(stepDecay)
    cs.add_hyperparameter(num_layers)
    cs.add_hyperparameter(num_units_0)
    cs.add_hyperparameter(num_units_1)
    cs.add_hyperparameter(num_units_2)
    cs.add_hyperparameter(num_units_3)
    cs.add_hyperparameter(SGD_momentum)
    cs.add_hyperparameter(sgd_initial)
    cs.add_hyperparameter(sgd_fraction)
    cs.add_hyperparameter(adam_initial)
    cs.add_hyperparameter(adam_fraction)
    cs.add_hyperparameter(dropout_0)
    cs.add_hyperparameter(dropout_1)
    cs.add_hyperparameter(dropout_2)
    cs.add_hyperparameter(dropout_3)
    cs.add_hyperparameter(l2_reg_0)
    cs.add_hyperparameter(l2_reg_1)
    cs.add_hyperparameter(l2_reg_2)
    cs.add_hyperparameter(l2_reg_3)
    cs.add_hyperparameter(activation)
    cs.add_hyperparameter(loss_function)
    cs.add_hyperparameter(optimizer)
    cs.add_hyperparameter(output_activation)
    cs.add_hyperparameter(learning_rate_schedule)
    # TODO: Implement the configuration space
    cond1 = CS.EqualsCondition(adam_fraction, optimizer, "Adam")
    cond2 = CS.EqualsCondition(adam_initial, optimizer, "Adam")
    cond3 = CS.EqualsCondition(sgd_initial, optimizer, "SGD")
    cond4 = CS.EqualsCondition(sgd_fraction, optimizer, "SGD")
    cond5 = CS.EqualsCondition(SGD_momentum, optimizer, "SGD")
    cond6 = CS.EqualsCondition(stepDecay, learning_rate_schedule, "StepDecay")
    cond7 = CS.EqualsCondition(dropout_3, num_layers, 4)
    cond8 = CS.EqualsCondition(l2_reg_3, num_layers, 4)
    cond9 = CS.EqualsCondition(num_units_3, num_layers, 4)
    cond10 = CS.GreaterThanCondition(dropout_2, num_layers, 3)
    cond11 = CS.GreaterThanCondition(dropout_1, num_layers, 2)
    cond12 = CS.GreaterThanCondition(l2_reg_2, num_layers, 3)
    cond13 = CS.GreaterThanCondition(l2_reg_1, num_layers, 2)
    cond14 = CS.GreaterThanCondition(num_units_1, num_layers, 2)
    cond15 = CS.GreaterThanCondition(num_units_2, num_layers, 3)
    cs.add_condition(cond1)
    cs.add_condition(cond2)
    cs.add_condition(cond3)
    cs.add_condition(cond4)
    cs.add_condition(cond5)
    cs.add_condition(cond6)
    cs.add_condition(cond7)
    cs.add_condition(cond8)
    cs.add_condition(cond9)
    cs.add_condition(cond10)
    cs.add_condition(cond11)
    cs.add_condition(cond12)
    cs.add_condition(cond13)
    cs.add_condition(cond14)
    cs.add_condition(cond15)
    return cs


def objective_function(config, epoch=127, **kwargs):
    # Cast the config to an array such that it can be forwarded
    # to the surrogate
    x = deepcopy(config.get_array())
    x[np.isnan(x)] = -1
    lc = rf.predict(x[None, :])[0]
    c = cost_rf.predict(x[None, :])[0]

    return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss, info = objective_function(cfg, epoch=int(budget))

        return ({
            'loss': loss,
            'info': {"runtime": info["cost"],
                     "lc": info["learning_curve"]}
        })


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_smac', action='store_true')
    parser.add_argument('--run_hyperband', action='store_true')
    parser.add_argument('--n_iters', default=50, type=int)
    args = vars(parser.parse_args())

    n_iters = args['n_iters']

    cs = create_config_space()
    rf = pickle.load(open("./rf_surrogate_paramnet_mnist.pkl", "rb"))
    cost_rf = pickle.load(open("./rf_cost_surrogate_paramnet_mnist.pkl", "rb"))

    if args["run_smac"]:
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": n_iters,
                             "cs": cs,
                             "deterministic": "true",
                             "output_dir": ""})

        smac = SMAC(scenario=scenario, tae_runner=objective_function)
        smac.optimize()

        # The following lines extract the incumbent strategy and the estimated
        # wall-clock time of the optimization
        rh = smac.runhistory
        incumbents = []
        incumbent_performance = []
        inc = None
        inc_value = 1
        idx = 1
        t = smac.get_trajectory()

        wall_clock_time = []
        cum_time = 0
        for d in rh.data:
            cum_time += rh.data[d].additional_info["cost"]
            wall_clock_time.append(cum_time)
        for i in range(n_iters):

            if idx < len(t) and i == t[idx].ta_runs - 1:
                inc = t[idx].incumbent
                inc_value = t[idx].train_perf
                idx += 1

            incumbents.append(inc)
            incumbent_performance.append(inc_value)

        # TODO: save and plot the wall clock time and the validation of the
        # incumbent after each iteration here

        lc_smac = []
        for d in rh.data:
            lc_smac.append(rh.data[d].additional_info["learning_curve"])
        plt.figure(1)
        plt.subplot(211)
        plt.plot(wall_clock_time, incumbent_performance, 'red')
        # lc_smac = list(map(list, zip(*lc_smac)))
        plt.figure(2)
        plt.subplot(211)
        for x in lc_smac:
            plt.plot(x, 'red')
        red_patch = mpatches.Patch(color='red', label='SMAC')
        # TODO: save and plot all learning curves here

    # if args["run_hyperband"]:
    if True:
        nameserver, ns = hpbandster.distributed.utils.start_local_nameserver()

        # starting the worker in a separate thread
        w = WorkerWrapper(nameserver=nameserver, ns_port=ns)
        w.run(background=True)

        CG = hpbandster.config_generators.RandomSampling(cs)

        # instantiating Hyperband with some minimal configuration
        HB = hpbandster.HB_master.HpBandSter(
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
        res = HB.run(10, min_n_workers=1)

        # shutdown the worker and the dispatcher
        HB.shutdown(shutdown_workers=True)

        # extract incumbent trajectory and all evaluated learning curves
        traj = res.get_incumbent_trajectory()
        wall_clock_time = []
        cum_time = 0

        for c in traj["config_ids"]:
            cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
            wall_clock_time.append(cum_time)

        lc_hyperband = []
        for r in res.get_all_runs():
            c = r["config_id"]
            lc_hyperband.append(res.get_runs_by_id(c)[-1]["info"]["lc"])
        incumbent_performance = traj["losses"]
        plt.figure(1)
        plt.subplot(212)
        plt.plot(wall_clock_time, incumbent_performance, 'green')
        # lc_hyperband = list(map(list, zip(*lc_hyperband)))
        plt.figure(2)
        plt.subplot(212)
        for x in lc_hyperband:
            plt.plot(x, 'red')

    green_patch = mpatches.Patch(color='green',
                                 label='Hyperband')
    plt.figure(1)
    plt.legend(handles=[red_patch, green_patch])
    plt.ylabel('Validation Error')
    plt.xlabel('Timesteps')
    plt.savefig('comparison.png')
    plt.figure(2)
    plt.ylabel('Learning Curves')
    plt.xlabel('Timesteps')
    plt.subplot(211)
    plt.legend(handles=[red_patch])
    plt.subplot(212)
    plt.legend(handles=[green_patch])
    plt.savefig('curves.png')
    # TODO: save and plot the wall clock time and the validation
    # of the incumbent after each iteration here
