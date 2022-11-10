import json
from pathlib import Path
from typing import Union
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry


class Client:
    """ navio API client

    Uploading data:

        >>> client = Client('https://navio.craftworks.io/', 'abc123')
        >>> client.upload_csv('./data.csv', 'workspace-id-1', 'my-data')

    Uploading models:

        >>> client = Client('https://navio.craftworks.io/', 'abc123')
        >>> client.upload_model_zip(
        ...     './model.zip', 'workspace-id-1',
        ...     'use-case-id-1', 'my-model')
    """

    def __init__(self, navio_url: str, api_token: str) -> 'Client':
        self._url = navio_url
        self._token = api_token
        self._session = self._create_session()
        self._session.headers.update({"Authorization": f"Bearer {api_token}"})

    @staticmethod
    def _create_session() -> requests.Session:
        """ Returns requests.Session configured with retries
        :return: configured requests session
        """
        session = requests.Session()
        retries = Retry(total=5,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _check_response(self, response: requests.Response) -> None:
        """ Checks response status, raising the response content if needed
        :param response: response object to check
        :return: None
        """
        try:
            response.raise_for_status()
        except Exception as err:
            raise type(err)(response.json()) from err

    def _resolve_url(self, suffix: str) -> str:
        return urljoin(self._url + '/', suffix)

    def assign_trainer_to_model(self, path: Union[str, Path],
                                model_id: str) -> None:
        suffix = f'/api/navio/v1/models/{model_id}/trainer-model'
        path = Path(path)
        with Path(path).open('rb') as file:
            files = [('trainer', (path.name, file, 'multipart/form-data'))]
            response = self._session.post(self._resolve_url(suffix),
                                          files=files)
        self._check_response(response)

    def assign_model_to_deployment(self, model_id: str,
                                   deployment_id: str) -> None:
        """ Assigns specified model to specified deployment

        Checking for deployment status is not required. Deployment is only
        usable if the model's status is "READY"

        :param model_id: uuid of the model
        :param deployment_id: uuid of the deployment
        :return: None
        """
        suffix = f'/api/navio/v1/deployments/{deployment_id}/models/{model_id}'
        response = self._session.put(self._resolve_url(suffix))
        self._check_response(response)

    def get_dataset_status(self, dataset_id: str, workspace_id: str) -> str:
        """ Returns status of the specified dataset in given workspace
        :param dataset_id: the uuid of the dataset
        :param workspace_id: the uuid of the workspace
        :return: status string in ['PROCESSING', 'READY', 'FAILED']
        """
        suffix = (f'/api/navio/v1/workspaces/{workspace_id}'
                  f'/datasets/file/{dataset_id}/status')
        response = self._session.get(self._resolve_url(suffix))
        self._check_response(response)
        return response.json()['value']

    def get_model_status(self, model_id: str) -> str:
        """ Returns status of the specified model
        :param model_id: the uuid of the model
        :return: status string in ['READY', 'DATASET_LOADING',
        'WAITING_FOR_CLUSTER', 'WAITING_FOR_RESOURCES', 'TRAINING_STARTED',
        'BUILDING_DOCKER_IMAGE', 'BUILDING_DOCKER_IMAGE_FAILED',
        'TRAINING_FAILED']
        """
        suffix = f'/api/navio/v1/models/{model_id}/status'
        response = self._session.get(self._resolve_url(suffix))
        self._check_response(response)
        return response.json()['state']

    def get_deployment_status(self, deployment_id: str) -> dict:
        """ Returns status of the specified deployment
        :param deployment_id: the uuid of the deployment
        :return: status info as a dictionary
        """
        suffix = f'/api/navio/v1/deployments/{deployment_id}/status'
        response = self._session.get(self._resolve_url(suffix))
        self._check_response(response)
        return response.json()

    def delete_model(self, model_id: str) -> str:
        """ Deletes the specified model
        :param model_id: the uuid of the model
        :return: None
        """
        suffix = f'/api/navio/v1/models/{model_id}'
        response = self._session.delete(self._resolve_url(suffix))
        self._check_response(response)

    def upload_model_zip(self, path: Union[Path, str], workspace_id: str,
                         use_case_id: str, name: str) -> str:
        """ Sends the model zip to navio. Returns model id
        :param path: local path to the model archive
        :param workspace_id: the uuid of the target workspace on navio
        :param use_case_id: the uuid of the target use case on navio
        :param name: the name to give the model
        :return: uuid of the created navio model
        """
        path = Path(path)
        suffix = (f'/api/navio/v1/models/workspaces/{workspace_id}'
                  f'/usecases/{use_case_id}/upload')

        dto_body = json.dumps({'name': name, 'description': 'description'})
        with path.open('rb') as file:
            # yapf: disable
            files = [
                ('createDTO', ('createDTO', dto_body, 'application/json')),
                ('model', (path.name, file, 'multipart/form-data'))
            ]
            # yapf: enable
            response = self._session.post(self._resolve_url(suffix),
                                          files=files)
        self._check_response(response)
        return response.json()['id']

    def upload_csv(self, path: Union[Path, str], workspace_id: str,
                   name: str) -> str:
        """ Sends the CSV file to navio. Returns dataset id
        :param path: local path to the csv dataset
        :param workspace_id: the uuid of the target workspace on navio
        :param name: the name to give the dataset
        :return: the uuid of the created navio dataset
        """
        suffix = f'/api/navio/v1/workspaces/{workspace_id}/datasets/file'
        dto_body = json.dumps({'name': name, 'columnDelimiter': 'COMMA'})
        with Path(path).open('rb') as file:
            # yapf: disable
            files = [
                ('createDTO', ('createDTO', dto_body, 'application/json')),
                ('file', ('file', file, 'multipart/form-data'))
            ]
            # yapf: enable
            response = self._session.post(self._resolve_url(suffix),
                                          files=files)
        self._check_response(response)
        return response.json()['id']

    def retrain_model(self, model_id: str, dataset_id: str) -> str:
        """ Starts the retraining of given model using given dataset
        :param model_id: the uuid of the model to retrain
        :param dataset_id: the uuid of the dataset to use for training
        :return: the uuid of the newly created model
        """
        suffix = f'/api/navio/v1/models/{model_id}/train'
        response = self._session.post(self._resolve_url(suffix),
                                      json={"datasetId": dataset_id})
        self._check_response(response)
        return response.json()['idOfNewModel']
