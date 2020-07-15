import logging

from trains import Task
from trains.automation import DiscreteParameterRange, HyperParameterOptimizer, RandomSearch, \
    UniformIntegerParameterRange, GridSearch

try:
    from trains.automation.hpbandster import OptimizerBOHB
    Our_SearchStrategy = GridSearch
except ValueError:
    logging.getLogger().warning(
        'Apologies, it seems you do not have \'hpbandster\' installed, '
        'we will be using RandomSearch strategy instead\n'
        'If you like to try ' '{{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},\n'
        'run: pip install hpbandster')
    Our_SearchStrategy = RandomSearch


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


task = Task.init(project_name='Roc-star Loss',
                 task_name='Automatic Hyper-Parameter Optimization',
                 task_type=Task.TaskTypes.optimizer,
                 reuse_last_task_id=False)

# MODIFY THIS, MAKE SURE TO CHOOSE THE CORRECT TEMPLATE TASK ID
# experiment template to optimize in the hyper-parameter optimization
args = {
    'template_task_id': '8d8ff6167a334ff59c3001c06e996eda',
    'run_as_service': False,
}
args = task.connect(args)

# Example use case:
an_optimizer = HyperParameterOptimizer(
    # This is the experiment we want to optimize
    base_task_id=args['template_task_id'],
    # here we define the hyper-parameters to optimize
    hyper_parameters=[
        DiscreteParameterRange('lstm_units', values=[64, 96, 128]),
        DiscreteParameterRange('dense_hidden_units', values=[512, 1024, 2048]),
        DiscreteParameterRange('use_roc_star', values=[True,False])
    ],
    # this is the objective metric we want to maximize/minimize
    objective_metric_title='Accuracy',
    objective_metric_series='validation accuracy',
    # now we decide if we want to maximize it or minimize it (accuracy we maximize)
    objective_metric_sign='max',
    # let us limit the number of concurrent experiments,
    # this in turn will make sure we do dont bombard the scheduler with experiments.
    # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
    max_number_of_concurrent_tasks=4,
    # this is the optimizer class (actually doing the optimization)
    # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
    # more are coming soon...
    optimizer_class=Our_SearchStrategy,
    # Select an execution queue to schedule the experiments for execution
    execution_queue='MY-EXPERIMENT-QUEUE',
    # Optional: Limit the execution time of a single experiment, in minutes.
    # (this is optional, and if using  OptimizerBOHB, it is ignored)
    #time_limit_per_job=10.,
    # Check the experiments every 6 seconds is way too often, we should probably set it to 5 min,
    # assuming a single experiment is usually hours...
    pool_period_min=0.1,
    # set the maximum number of jobs to launch for the optimization, default (None) unlimited
    # If OptimizerBOHB is used, it defined the maximum budget in terms of full jobs
    # basically the cumulative number of iterations will not exceed total_max_jobs * max_iteration_per_job
    total_max_jobs=10,
    # This is only applicable for OptimizerBOHB and ignore by the rest
    # set the minimum number of iterations for an experiment, before early stopping
    min_iteration_per_job=10,
    # Set the maximum number of iterations for an experiment to execute
    # (This is optional, unless using OptimizerBOHB where this is a must)
    max_iteration_per_job=30,
)

# if we are running as a service, just enqueue ourselves into the services queue and let it run the optimization
if args['run_as_service']:
    # if this code is executed by `trains-agent` the function call does nothing.
    # if executed locally, the local process will be terminated, and a remote copy will be executed instead
    task.execute_remotely(queue_name='services', exit_process=True)

# report every 12 seconds, this is way too often, but we are testing here J
an_optimizer.set_report_period(2.2)
# start the optimization process, callback function to be called every time an experiment is completed
# this function returns immediately
an_optimizer.start(job_complete_callback=job_complete_callback)
# set the time limit for the optimization process (2 hours)
an_optimizer.set_time_limit(in_minutes=120.0)
# wait until process is done (notice we are controlling the optimization process in the background)
an_optimizer.wait()
# optimization is completed, print the top performing experiments id
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
# make sure background optimization stopped
an_optimizer.stop()

print('We are done, good bye')
