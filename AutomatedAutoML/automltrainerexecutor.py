import pandas as pd
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Run

def azureml_main(dataframe1 = None, dataframe2 = None):
    experiment_name = '…'

    run = Run.get_context(allow_offline=True)
    ws = run.experiment.workspace

    experiment = Experiment(ws, experiment_name)
    config = ScriptRunConfig(source_directory='./Script Bundle', script='automltrainer.py', compute_target='…')

    new_run = experiment.submit(config)

    return dataframe1,
