Error description
====

When using the following command line to start the deployment of a navio model locally:

.. code-block:: bash

    mlflow models build-docker -m ./model --install-mlflow -n mlflow-model-image

I obtained the following error:

.. code-block:: bash

    File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/transport/unixconn.py", line 27, in connect
        sock.connect(self.unix_socket)
    FileNotFoundError: [Errno 2] No such file or directory

    [...]

    File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/transport/unixconn.py", line 27, in connect
        sock.connect(self.unix_socket)
    urllib3.exceptions.ProtocolError: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))

    [...]

    File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/api/client.py", line 221, in _retrieve_server_version
        raise DockerException(
    docker.errors.DockerException: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))


.. raw:: html

   <details>
       <summary >Click to reveal the complete error details</summary>

        <pre>
    <xmp>
    (pynavio) userhome@users-MBP test_models % mlflow models build-docker -m ./timeseries_trainer_model --install-mlflow -n mlflow-model-image
    Downloading artifacts: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1068.34it/s]
    2023/11/27 17:26:57 INFO mlflow.models.flavor_backend_registry: Selected backend for flavor 'python_function'
    Downloading artifacts:   0%|                                                                                                     | 0/46 [00:00<?, ?it/s]2023/11/27 17:26:57 INFO mlflow.store.artifact.artifact_repo: The progress bar can be disabled by setting the environment variable MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR to false
    Downloading artifacts: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 46/46 [00:00<00:00, 6367.59it/s]
    2023/11/27 17:26:57 INFO mlflow.models.docker_utils: Building docker image with name mlflow-model-image
    Traceback (most recent call last):
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/connectionpool.py", line 790, in urlopen
        response = self._make_request(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/connectionpool.py", line 496, in _make_request
        conn.request(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/connection.py", line 395, in request
        self.endheaders()
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/http/client.py", line 1247, in endheaders
        self._send_output(message_body, encode_chunked=encode_chunked)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/http/client.py", line 1007, in _send_output
        self.send(msg)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/http/client.py", line 947, in send
        self.connect()
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/transport/unixconn.py", line 27, in connect
        sock.connect(self.unix_socket)
    FileNotFoundError: [Errno 2] No such file or directory

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/requests/adapters.py", line 486, in send
        resp = conn.urlopen(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/connectionpool.py", line 844, in urlopen
        retries = retries.increment(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/util/retry.py", line 470, in increment
        raise reraise(type(error), error, _stacktrace)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/util/util.py", line 38, in reraise
        raise value.with_traceback(tb)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/connectionpool.py", line 790, in urlopen
        response = self._make_request(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/connectionpool.py", line 496, in _make_request
        conn.request(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/urllib3/connection.py", line 395, in request
        self.endheaders()
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/http/client.py", line 1247, in endheaders
        self._send_output(message_body, encode_chunked=encode_chunked)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/http/client.py", line 1007, in _send_output
        self.send(msg)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/http/client.py", line 947, in send
        self.connect()
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/transport/unixconn.py", line 27, in connect
        sock.connect(self.unix_socket)
    urllib3.exceptions.ProtocolError: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/api/client.py", line 214, in _retrieve_server_version
        return self.version(api_version=False)["ApiVersion"]
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/api/daemon.py", line 181, in version
        return self._result(self._get(url), json=True)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/utils/decorators.py", line 46, in inner
        return f(self, *args, **kwargs)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/api/client.py", line 237, in _get
        return self.get(url, **self._set_request_timeout(kwargs))
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/requests/sessions.py", line 602, in get
        return self.request("GET", url, **kwargs)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/requests/adapters.py", line 501, in send
        raise ConnectionError(err, request=request)
    requests.exceptions.ConnectionError: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
      File "/Users/userhome/miniconda3/envs/pynavio/bin/mlflow", line 8, in <module>
        sys.exit(cli())
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/click/core.py", line 1157, in __call__
        return self.main(*args, **kwargs)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/click/core.py", line 1078, in main
        rv = self.invoke(ctx)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/click/core.py", line 1688, in invoke
        return _process_result(sub_ctx.command.invoke(sub_ctx))
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/click/core.py", line 1688, in invoke
        return _process_result(sub_ctx.command.invoke(sub_ctx))
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/click/core.py", line 1434, in invoke
        return ctx.invoke(self.callback, **ctx.params)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/click/core.py", line 783, in invoke
        return __callback(*args, **kwargs)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/mlflow/models/cli.py", line 267, in build_docker
        build_docker_api(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/mlflow/models/__init__.py", line 80, in build_docker
        get_flavor_backend(model_uri, docker_build=True, env_manager=env_manager).build_image(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/mlflow/pyfunc/backend.py", line 350, in build_image
        _build_image(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/mlflow/models/docker_utils.py", line 221, in _build_image
        _build_image_from_context(context_dir=cwd, image_name=image_name)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/mlflow/models/docker_utils.py", line 227, in _build_image_from_context
        client = docker.from_env()
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/client.py", line 96, in from_env
        return cls(
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/client.py", line 45, in __init__
        self.api = APIClient(*args, **kwargs)
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/api/client.py", line 197, in __init__
        self._version = self._retrieve_server_version()
      File "/Users/userhome/miniconda3/envs/pynavio/lib/python3.8/site-packages/docker/api/client.py", line 221, in _retrieve_server_version
        raise DockerException(
    docker.errors.DockerException: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))

        </xmp>
        </pre>

   </details>





Information
____
Information:

- Computer: MacOs
- Virtual environment: conda
- Python version: Python 3.10.13

Issue description:
It is an issue related to the latest release of docker, where the context of the client is changed from ``default`` to ``desktop-linux`` which uses different endpoint and therefore breaks the docker client.


Solution
____

In order to solve the problem the following steps need to be followed:

#. Run the following command → Check that the client is in the desktop one and not in the default. This is the issue and what needs to be changed.

   .. code-block:: bash

      $ docker context ls
      NAME                TYPE                DESCRIPTION                               DOCKER ENDPOINT                                  KUBERNETES ENDPOINT   ORCHESTRATOR
      default             moby                Current DOCKER_HOST based configuration   unix:///var/run/docker.sock                                            swarm
      desktop-linux *     moby                                                          unix:///Users/ec2-user/.docker/run/docker.sock

#. Apply one of the possible solutions

   - Temporal solution → You can use one of the following command lines

      .. code-block:: bash

         export DOCKER_HOST=<endpoint of default context>
         # or
         docker context use default

   - Definitive solution → Run the following command line

      .. code-block:: bash

         sudo ln -s "$HOME/.docker/run/docker.sock" /var/run/docker.sock

#. Run again the firs command and check that the (*) has changed to the ``default`` one.

Theoretically, each time you open a new terminal you will have to follow the same steps if you chose the temporal solution. However, for me I did it once and now it always works.
