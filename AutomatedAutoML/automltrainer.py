import os
os.system(f"pip install azureml-train-automl-client==1.22.0")
import pandas as pd
import azureml.core
from azureml.core import Experiment, Workspace, Run
from azureml.core.dataset import Dataset
from azureml.core.compute import ComputeTarget
from azureml.core.authentication import MsiAuthentication
from azureml.train.automl import AutoMLConfig

tenant_id = '...'
resource_group = '...'
workspace_name = '...'
msi_identity_config = {"client_id": "..."}

dataset_name = 'auto-coffee-consumption'
dataset_label_column = 'Coffee consumption'
experiment_name = 'automated-auto-coffee-consumption'
compute_name = 'jod-cc'

msi_auth = MsiAuthentication(identity_config=msi_identity_config)
ws = Workspace(subscription_id=tenant_id,
               resource_group=resource_group,
               workspace_name=workspace_name,
               auth=msi_auth)
			   
dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
compute_target = ws.compute_targets[compute_name]

automl_config = AutoMLConfig(task='regression',
                             experiment_timeout_minutes=15,
                             primary_metric='normalized_root_mean_squared_error',
                             training_data=dataset,
							 compute_target=compute_target,
                             label_column_name=dataset_label_column)

experiment = Experiment(ws, experiment_name)

run = experiment.submit(automl_config, show_output=True)
run.wait_for_completion()
