# Run black on .py files
black exercise1.py solution2.py solution3.py solution_bonus1.py


# Convert .py to ipynb
# "cell_metadata_filter": "all" preserve cell tags including our solution tags
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' exercise1.py --output ../exercise1.ipynb
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' solution2.py --output ../solution2.ipynb
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' solution3.py --output ../solution3.ipynb
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' solution_bonus1.py --output ../solution_bonus1.ipynb

# Create the exercise notebook by removing cell outputs and deleting cells tagged with "solution"
# There is a bug in the nbconvert cli so we need to use the python API instead
python convert-solution.py ../solution2.ipynb ../exercise2.ipynb
python convert-solution.py ../solution3.ipynb ../exercise3.ipynb
python convert-solution.py ../solution_bonus1.ipynb ../exercise_bonus1.ipynb

