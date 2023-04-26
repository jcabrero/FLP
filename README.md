# Automated Parameter Selection based on Fuzzy Logic and Linear Programming

This repository covers the code used for the creation of the code described in:

## Using this repository

There are three ways to use the repository:

### Devcontainer

Using Github Codespaces, you can automatically setup an environment to use the repository.


### Docker
The command below will open a terminal in the container.
```
cd docker
make run
```

Then navigate run `jupyterlab`:

```
cd scripts
bash jupyterlab.sh
```

### Python Requirements

```
pip install -r requirements.txt
```

## Citation

```
@article{cabrero2023towards,
  title={Towards Automated Homomorphic Encryption Parameter Selection with Fuzzy Logic and Linear Programming},
  author={Cabrero-Holgueras, Jos{\'e} and Pastrana, Sergio},
  journal={arXiv preprint arXiv:2302.08930},
  year={2023}
}
```

## LICENSE
The code is released without any guarantees under a [GPLv3 License](/LICENSE).Any copy or modification must be released under the same license.