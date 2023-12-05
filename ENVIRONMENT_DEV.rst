Error description
====

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
    .. code-block::
        $ docker context ls
        NAME                TYPE                DESCRIPTION                               DOCKER ENDPOINT                                  KUBERNETES ENDPOINT   ORCHESTRATOR
        default             moby                Current DOCKER_HOST based configuration   unix:///var/run/docker.sock                                            swarm
        desktop-linux *     moby                                                          unix:///Users/ec2-user/.docker/run/docker.sock

#. Apply one of the possible solutions
    - Temporal solution → You can use one of the following command lines
        .. code-block::
            export DOCKER_HOST=<endpoint of default context>
            or
            docker context use default

    - Definitive solution → Run the following command line
        .. code-block::
            sudo ln -s "$HOME/.docker/run/docker.sock" /var/run/docker.sock

#. Run again the firs command and check that the (*) has changed to the ``default`` one.

Theoretically, each time you open a new terminal you will have to follow the same steps if you chose the temporal solution. However, for me I did it once and now it always works.
