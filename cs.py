import ConfigSpace as CS

def create_config_space():
    # configuration space for deep4.py
    cs = CS.ConfigurationSpace()

    n_filters_time = CS.UniformIntegerHyperparameter("n_filters_time",
                                                 lower=20,
                                                 default_value=25,
                                                 upper=70)
    n_filters_spat = CS.UniformIntegerHyperparameter("n_filters_time",
                                                  lower=20,
                                                  default_value=25,
                                                  upper=70,
                                                  log=False)
    filter_time_length = CS.UniformIntegerHyperparameter("filter_time_length",
                                                lower=3,
                                                default_value=10,
                                                upper=25,
                                                log=False)
    pool_time_length = CS.UniformIntegerHyperparameter("pool_time_length",
                                                  lower=1,
                                                  default_value=3,
                                                  upper=8,
                                                  log=False)
    pool_time_stride = CS.UniformIntegerHyperparameter("pool_time_stride",
                                                  lower=1,
                                                  default_value=3,
                                                  upper=8,
                                                  log=False)
    n_filters_2 = CS.UniformIntegerHyperparameter("n_filters_2",
                                                  lower=8,
                                                  default_value=50,
                                                  upper=265,
                                                  log=True)
    fliter_length_2 = CS.UniformIntegerHyperparameter("filter_length_2",
                                                  lower=3,
                                                  default_value=10,
                                                  upper=14,
                                                  log=False)
    n_filters_3 = CS.UniformIntegerHyperparameter("n_filters_3",
                                                  lower=8,
                                                  default_value=100,
                                                  upper=265,
                                                  log=True)
    fliter_length_3 = CS.UniformIntegerHyperparameter("filter_length_3",
                                                  lower=3,
                                                  default_value=10,
                                                  upper=14,
                                                  log=False)
    n_filters_4 = CS.UniformIntegerHyperparameter("n_filters_4",
                                                  lower=8,
                                                  default_value=200,
                                                  upper=265,
                                                  log=True)
    fliter_length_4 = CS.UniformIntegerHyperparameter("filter_length_4",
                                                  lower=3,
                                                  default_value=10,
                                                  upper=14,
                                                  log=False)
    first_nonlin = CS.CategoricalHyperparameter("first_nonlin", ['elu', 'relu6', 'relu', 'tanh', 'square'])
    first_pool_mode = CS.CategoricalHyperparameter("first_pool_mode", ['max', 'mean'])
    first_pool_nonlin = CS.CategoricalHyperparameter("first_pool_nonlin", ['identity', 'log'])
    later_nonlin = CS.CategoricalHyperparameter("later_nonlin", ['elu', 'relu6', 'relu', 'tanh', 'square'])
    later_pool_mode = CS.CategoricalHyperparameter("later_pool_mode", ['max', 'mean'])
    later_pool_nonlin = CS.CategoricalHyperparameter("later_pool_nonlin", ['identity', 'log'])
    drop_prob = CS.UniformFloatHyperparameter("drop_prob",
                                                  lower=0.0,
                                                  default_value=0.5,
                                                  upper=0.9,
                                                  log=False)
    split_first_layer = CS.CategoricalHyperparameter("split_first_layer",['True', 'False'] )
    batch_norm = CS.CategoricalHyperparameter("batch_norm",['True', 'False'] )
    stride_before_pool = CS.CategoricalHyperparameter("stride_before_pool",['False', 'True'] )
    batch_norm_alpha = CS.UniformFloatHyperparameter("batch_norm_alpha",
                                                  lower=0.01,
                                                  default_value=0.1,
                                                  upper=0.9,
                                                  log=False)
    cs.add_hyperparameter(n_filters_time)
    cs.add_hyperparameter(filter_time_length)
    cs.add_hyperparameter(pool_time_length)
    cs.add_hyperparameter(pool_time_stride)
    cs.add_hyperparameter(n_filters_2)
    cs.add_hyperparameter(filter_length_2)
    cs.add_hyperparameter(n_filters_3)
    cs.add_hyperparameter(filter_length_3)
    cs.add_hyperparameter(n_filters_4)
    cs.add_hyperparameter(filter_length_4)
    cs.add_hyperparameter(first_nonlin)
    cs.add_hyperparameter(first_pool_mode)
    cs.add_hyperparameter(first_pool_nonlin)
    cs.add_hyperparameter(later_pool_mode)
    cs.add_hyperparameter(later_pool_nonlin)
    cs.add_hyperparameter(later_nonlin)
    cs.add_hyperparameter(drop_prob)
    cs.add_hyperparameter(split_first_layer)
    cs.add_hyperparameter(batch_norm)
    cs.add_hyperparameter(stride_before_pool)

	
    cond1 = CS.EqualsCondition(input_time_length, final_conv_length, "?")
    cond2 = CS.EqualsCondition(n_filters_spat, split_first_layer, "True")
    cond3 = CS.EqualsCondition(batch_norm_alpha, batch_norm, "True")

    cs.add_condition(cond1)
    cs.add_condition(cond2)
    cs.add_condition(cond3)
	
    return cs
