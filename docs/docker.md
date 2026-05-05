# Docker

If you prefer not to install dependencies locally, or if you want to train your models on a containerized remote machine, you can use the provided Dockerfile to build an image with all dependencies pre-installed.

The only prerequisites are [Docker](https://docs.docker.com/get-docker/) and, on your deployment machine, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support.

To build the Docker image, run the following command from the root of the repository:

```bash
docker build -f Dockerfile -t r2dreamer:local .
```
You can replace the `-t` argument with any image name you like. The command above will build and tag the image as `r2dreamer:local`.

Then start a container from the built image with:

```bash
docker run -it -d --rm \
    --gpus=all \
    --network=host \
    --volume=$PWD:/workspace \
    --name=r2dreamer-container \
    r2dreamer:local
```

You can then connect to the running container and execute your training scripts. For example, to run R2-Dreamer on DMC Walker Walk:

```bash
# Connect to the running container
docker exec -it r2dreamer-container bash

# And then inside the container:
python3 train.py env=dmc_vision env.task=dmc_walker_walk

# Alternatively, combine it with the docker exec command and use the -d flag to run in detached mode:
docker exec -it -d r2dreamer-container bash -c "python3 train.py env=dmc_vision env.task=dmc_walker_walk"
```

Training metrics are logged to [Weights & Biases](https://wandb.ai). Pass your API key into the container so the run can stream metrics out:

```bash
docker run -it -d --rm \
    --gpus=all \
    --network=host \
    --volume=$PWD:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --name=r2dreamer-container \
    r2dreamer:local
```

Runs then show up in the `dreamerv3-backbones` project on wandb (override via `wandb.project=...` on the CLI).

> Docker documentation contributed by [@MeierTobias](https://github.com/MeierTobias).
