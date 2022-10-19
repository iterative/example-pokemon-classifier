# Jupyter Notebook for Pokémon type classifier 

This Jupyter Notebook trains a CNN to classify images of Pokémon. It will
predict whether a Pokémon is of a predetermined type (default: water). It is a
starting point that shows how a notebook might look before it is transformed
into a DVC pipeline.

_Note: due to the limited size of the dataset, the evaluation dataset is the
same data set as the train+test. Take the results of the model with a grain of
salt._

## From Notebook to pipeline

This project details the transformation from Notebook to DVC pipeline. In the
different branches, you can find three stages in this process:

- [`snapshot-jupyter`](https://github.com/iterative/example-pokemon-classifier/tree/snapshot-jupyter):
  a prototype as you might build it in a Jupyter Notebook
- [`papermill-dvc`](https://github.com/iterative/example-pokemon-classifier/tree/papermill-dvc):
  a DVC pipeline with a single stage to run a parameterized
  notebook using [Papermill](https://papermill.readthedocs.io/)
- [`dvc-pipeline`](https://github.com/iterative/example-pokemon-classifier/tree/dvc-pipeline):
  pure DVC pipeline with Python modules

## Requirements

- [Python >= 3.9.13](https://www.python.org/downloads/)
- [Virtualenv >= 20.14.1](https://virtualenv.pypa.io/en/latest/installation.html)

## How to run

1. Create a new virtual environment with `virtualenv -p python3 .venv`

2. Activate the virtual environment with `source .venv/bin/activate`

3. Install the dependencies with `pip install -r requirements.txt`

4. Download the datasets from Kaggle into the data/external/ directory.

   ```console
   $ wget https://www.kaggle.com/datasets/robdewit/pokemon-images -o data/external/pokemon-gen-1-8
   $ wget https://www.kaggle.com/datasets/rounakbanik/pokemon -o data/external/stats/pokemon-gen-1-8.csv
   ```

5. Initialize DVC with `dvc init`

6. Start versioning the contents of the `data` directory with `dvc add data`

7. Launch the Notebook with `jupyter-notebook`

## Notes on hardware

The requirements specify `tensorflow-macos` and `tensorflow-metal`, which are
the appropriate requirements when you are using a Mac with an M1 CPU or later.
In case you are using a different system, you will need to replace these with
`tensorflow`.

