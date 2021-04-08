os.system(f"pip install azureml-train-automl-client==1.24.0")
import pandas as pd
import requests
import azureml.core
from azureml.core import Experiment, Workspace, Run
from azureml.core.dataset import Dataset
from azureml.core.compute import ComputeTarget
from azureml.core.authentication import MsiAuthentication
from azureml.train.automl import AutoMLConfig
from azureml.core.model import Model
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.conda_dependencies import CondaDependencies

#Azure Machine Learning workspace settings
subscription_id = '…'
resource_group = '…'
workspace_name = '…'
msi_identity_config = {"client_id": "…"}

#AutoML run settings
dataset_name = '…'
dataset_label_column = '…'
experiment_name = '…'
compute_name = '…'

#Model registration and Docker container publish settings
model_name = '…'
docker_image_name = '…'
docker_image_label = 'latest'
docker_webhook_url = '…'

#Retrieve the Azure ML Workspace
msi_auth = MsiAuthentication(identity_config=msi_identity_config)
ws = Workspace(subscription_id=subscription_id,
               resource_group=resource_group,
               workspace_name=workspace_name,
               auth=msi_auth)

#Retrieve dataset and compute for the AutoML run
dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
compute_target = ws.compute_targets[compute_name]

automl_config = AutoMLConfig(task='regression',
                             experiment_timeout_minutes=30,
                             primary_metric='normalized_root_mean_squared_error',
                             training_data=dataset,
							 compute_target=compute_target,
                             label_column_name=dataset_label_column)

#Execute the AutoML run
experiment = Experiment(ws, experiment_name)
run = experiment.submit(automl_config, show_output=True)
run.wait_for_completion()

#Get the best model from the AutoML run and register it
best_run = run.get_best_child()
best_run.download_files(prefix='outputs', append_prefix=False)
model = Model.register(model_path='outputs/model.pkl',
                       model_name=model_name,
                       workspace=ws)

#Prepare an environment for the model
myenv = Environment.from_conda_specification(name='project_environment', file_path='outputs/conda_env_v_1_0_0.yml')
myenv.docker.enabled = True
inference_config = InferenceConfig(entry_script='outputs/scoring_file_v_1_0_0.py', environment=myenv)

#Create Docker container for the model
package = Model.package(ws, [model], inference_config,
					   image_name=docker_image_name,
					   image_label=docker_image_label)
package.wait_for_creation(show_output=True)

#Update web app with the latest container
requests.post(docker_webhook_url)
