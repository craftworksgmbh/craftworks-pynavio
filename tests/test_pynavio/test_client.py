import socket
from contextlib import closing, contextmanager
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Generator
from wsgiref.simple_server import WSGIRequestHandler, make_server

import pytest

from pynavio.client import Client


def _find_free_port() -> int:
    """ Use the OS to get an unused port
    """
    with closing(socket.socket()) as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


MOCK_SERVER_PORT = _find_free_port()
MOCK_SERVER_URL = f'http://localhost:{MOCK_SERVER_PORT}'
MOCK_SERVER_TOKEN = 'abc123'


class QuietRequestHandler(WSGIRequestHandler):

    def log_message(self, *args, **kwargs) -> None:
        """ Empty so that the server doesn't log every request """
        pass


@contextmanager
def _request_capture(response: str = '') -> Generator:
    """ Accepts requests, pushing their payload onto the returned queue
    :param response: what the server should respond with
    :return: reference to the queue
    """
    queue = Queue()

    def _handle(environ, start_response) -> list:
        length = int(environ.get('CONTENT_LENGTH', '') or 0)
        # yapf: disable
        queue.put_nowait({
            'content': environ['wsgi.input'].read(length),
            'auth': environ.get('HTTP_AUTHORIZATION'),
            'path': environ.get('PATH_INFO'),
            'query': environ.get('QUERY_STRING')
        })
        # yapf: enable
        start_response('200 OK', [('Content-type', 'text/plain')])
        return [response.encode()]

    _server = make_server('0.0.0.0',
                          MOCK_SERVER_PORT,
                          _handle,
                          handler_class=QuietRequestHandler)
    try:
        thread = Thread(target=_server.serve_forever)
        thread.start()
        yield queue
    finally:
        _server.shutdown()
        thread.join()


@pytest.fixture
def client() -> Client:
    return Client(MOCK_SERVER_URL, api_token=MOCK_SERVER_TOKEN)


def test_get_dataset_status(client: Client) -> None:
    dataset_id = 'fake-dataset-id'
    workspace_id = 'fake-workspace-id'

    expected_value = 'ok'

    with _request_capture(f'{{"value": "{expected_value}"}}') as capture:
        result = client.get_dataset_status(dataset_id, workspace_id)
        assert result == expected_value

        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'

        assert dataset_id in payload.get('path')
        assert workspace_id in payload.get('path')


def test_get_model_status(client: Client) -> None:
    model_id = 'fake-model-id'

    expected_state = 'ok'

    with _request_capture(f'{{"state": "{expected_state}"}}') as capture:
        result = client.get_model_status(model_id)
        assert result == expected_state

        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'

        assert model_id in payload.get('path')


def test_get_deployment_status(client: Client) -> None:
    deployment_id = 'fake-deployment-id'

    with _request_capture('{"oodStatus": "MAJOR"}') as capture:
        result = client.get_deployment_status(deployment_id)
        assert result == dict(oodStatus='MAJOR')

        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'

        assert deployment_id in payload.get('path')


def test_assign_model_to_deployment(client: Client) -> None:
    model_id = 'fake-model-id'
    deployment_id = 'fake-deployment-id'

    with _request_capture() as capture:
        client.assign_model_to_deployment(model_id, deployment_id)

        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'

        assert model_id in payload.get('path')
        assert deployment_id in payload.get('path')


def test_assign_trainer_to_model(client: Client, tmp_path: Path) -> None:
    trainer_content = 'fake-trainer-content'
    model_id = 'fake-model-id'

    path = tmp_path / 'trainer.zip'
    with path.open('w') as file:
        file.write(trainer_content)

    with _request_capture() as capture:
        client.assign_trainer_to_model(path, model_id)

        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'

        assert model_id in payload.get('path')
        assert trainer_content.encode() in payload.get('content')


def test_upload_model_zip(client: Client, tmp_path: Path) -> None:
    use_case_id = 'fake-use-case-id'
    workspace_id = 'fake-workspace-id'
    model_name = 'fake-model-name'
    model_content = 'fake-model-content'

    path = tmp_path / 'model.zip'
    with path.open('w') as file:
        file.write(model_content)

    with _request_capture('{"id": "some-id"}') as capture:
        client.upload_model_zip(path, workspace_id, use_case_id, model_name)

        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'

        assert use_case_id in payload.get('path')
        assert workspace_id in payload.get('path')

        assert model_name.encode() in payload.get('content')
        assert model_content.encode() in payload.get('content')


def test_upload_csv(client: Client, tmp_path: Path) -> None:
    workspace_id = 'fake-workspace-id'
    csv_name = 'fake-csv-name'
    csv_content = 'fake-csv-content'

    path = tmp_path / 'data.csv'
    with path.open('w') as file:
        file.write(csv_content)

    with _request_capture('{"id": "some-id"}') as capture:
        client.upload_csv(path, workspace_id, csv_name)

        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'

        assert workspace_id in payload.get('path')

        assert csv_name.encode() in payload.get('content')
        assert csv_content.encode() in payload.get('content')


def test_retrain_model(client: Client) -> None:
    dataset_id = 'fake-dataset-id'
    model_id = 'fake-model-id'
    with _request_capture('{"idOfNewModel": "some-id"}') as capture:
        client.retrain_model(model_id, dataset_id)
        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'
        assert model_id in payload.get('path')
        assert dataset_id.encode() in payload.get('content')


def test_delete_model(client: Client) -> None:
    model_id = 'fake-model-id'

    with _request_capture() as capture:
        client.delete_model(model_id)

        payload = capture.get()
        assert payload.get('auth') == f'Bearer {MOCK_SERVER_TOKEN}'

        assert model_id in payload.get('path')
