## Environment setup

For benchmarking purpose, individual python environments are created.

- python=3.12
- CUDA 12
- ptyrad/quantem are using pytorch 2.7.1
- phaser is pulling the latest Jax for now (0.9.1 as of 2026.03.07)

cd into `ptycho_benchmark/`, then run the following commands:
```sh
./envs/create_ptyrad_env.sh
./envs/create_phaser_env.sh
./envs/create_quantem_env.sh --local
```

Each shell script will clear the pip/conda cache, create python environment (i.e., `bench_ptyrad`), and install corresponding packages via PyPI by default.

If the environment already exists, you can pass `--force` to remove it cleanly and create a new one.

Use `--local` to install the package from existing local source codes under `../ptyrad/`.

Use `--git <GIT_REF>` to install the package from git references, like `./create_ptyrad_env.sh --git v0.1.0.b13-post1`

Note that `quantem` currently (`dev`@787585e, 20260306) doesn't have a timing feature, so a locally modified version is used.