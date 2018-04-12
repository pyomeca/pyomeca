- write test suite for fileio and rototrans
- add write_csv to Markers3d and Analogs3d
- add write_csv test
- write examples for fileio

- create `env.yml` and `env_dev.yml` (replace Pipfile)
    - to install dependencies in a clean conda env, run: `conda env create -f env.yml`
    - to install all dependencies (included dev) in a clean conda env, run: `conda env create -f env_dev.yml`

- to run test with code coverage
    ```bash
    cd Documents/codes/pyomeca
    conda activate pyomeca
    python -m pytest --cov=pyomeca tests/
    ```