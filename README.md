# DVC pipeline for Pokémon type classifier 

# How to run
1. Create a new virtual environment with `virtualenv -p python3 envname`
2. Activate the virtual environment with `source .venv/bin/activate`
3. Install the dependencies with `pip install -r requirements.txt`
4. Add the DVC remote with `dvc remote add -d gdrive gdrive://1WRnCJzZhPRdtoM9Y3XtrGpIotqTy-UXZ`
5. Pull the data from the DVC remote with `dvc pull`
6. Launch the notebook with `jupyter-notebook`
