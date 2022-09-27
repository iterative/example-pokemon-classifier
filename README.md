# Jupyter Notebook for Pokémon type classifier 

This Jupyter Notebook trains a CNN to classify images of Pokémon. It will
predict whether a Pokémon is of a predetermined type (default: water). It is a
starting point that shows how a notebook might look before it is transformed
into a DVC pipeline.

_Note: due to the limited size of the dataset, the evaluation dataset is the
same data set as the train+test. Take the results of the model with a grain of
salt._

# From Notebook to pipeline

This project details the transformation from Notebook to DVC pipeline. In the
different branches, you can find three stages in this process:

- `snapshot-jupyter`: a prototype as you might build it in a Jupyter Notebook
- `papermill-dvc`: a DVC pipeline with a single stage to run a parameterized
  notebook using [Papermill](https://papermill.readthedocs.io/)
- `dvc-pipeline`: pure DVC pipeline with Python modules

# How to run
1. Create a new virtual environment with `virtualenv -p python3 .venv`
2. Activate the virtual environment with `source .venv/bin/activate`
3. Install the dependencies with `pip install -r requirements.txt`
4. Download the datasets from Kaggle:
  - `data/external/pokemon-gen-1-8`:
    https://www.kaggle.com/datasets/robdewit/pokemon-images
  - `data/external/stats/pokemon-gen-1-8.csv`:
    https://www.kaggle.com/datasets/rounakbanik/pokemon
5. Launch the Notebook with `jupyter-notebook`
