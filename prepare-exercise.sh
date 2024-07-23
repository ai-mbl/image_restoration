# Run black on .py files
black 01_CARE/care_solution.py 02_Noise2Void/n2v_solution.py 03_COSDD/solution-1-training.py 03_COSDD/solution-2-prediction.py 03_COSDD/solution-bonus-generation.py 04_bonus_denoiSplit/bonus_denoisplit.py 04_bonus_Noise2Noise/n2n_solution.py 

# Convert .py to ipynb
# "cell_metadata_filter": "all" preserve cell tags including our solution tags
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 01_CARE/care_solution.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 02_Noise2Void/n2v_solution.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 03_COSDD/solution-1-training.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 03_COSDD/solution-2-prediction.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 03_COSDD/solution-bonus-generation.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 04_bonus_denoiSplit/bonus_denoisplit.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 04_bonus_Noise2Noise/n2n_solution.py

# Create the exercise notebook by removing cell outputs and deleting cells tagged with "solution"
# There is a bug in the nbconvert cli so we need to use the python API instead
python convert-solution.py 01_CARE/care_solution.py 01_CARE/care_exercise.py
python convert-solution.py 02_Noise2Void/n2v_solution.py 02_Noise2Void/n2v_exercise.py
python convert-solution.py 03_COSDD/solution-1-training.py 03_COSDD/exercise-1-training.py
python convert-solution.py 03_COSDD/solution-2-prediction.py 03_COSDD/exercise-2-prediction.py
python convert-solution.py 03_COSDD/solution-bonus-generation.py 03_COSDD/exercise-bonus-generation.py
python convert-solution.py 04_bonus_Noise2Noise/n2n_solution.py 04_bonus_Noise2Noise/n2n_exercise.py