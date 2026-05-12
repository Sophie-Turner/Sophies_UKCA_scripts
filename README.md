Here are Sophie's Python scripts for the PhD work. Most are for processing UKCA data and working with machine learning models and their data. 
All of the scripts were written to run on Cambridge chemistry department's atmospheric servers, except for .ipynb files, which are notebooks that can be run in Colab on a browser, and scripts in the 'jasmin' directory, which are specifically to be run on the JASMIN supercomputer.
Contact Sophie Turner at st838@cam.ac.uk for more info.
If unavailable, contact Luke Abraham at nla
The supplied conda evironment, cenv, can be used for all Python scripts except the flat_fields scripts.  

There are several things you can do to repeat my work:
A. Output the UKCA & UM data you need to make ML training data.
B. Preprocess UKCA output data to use as ML training inputs.
C. Repeat model comparisons experiments.
D. Train a random forest photolysis emulator for use in UKCA.
E. Integrate the emulator into UKCA & UM version 13.9.
You don't need to do all of them. I have some saved datasets and models you can download and use to avoid some of these parts if you wish. See the instructions below for whichever task you need to do.



Part A: How to output the UKCA & UM data you need to make ML training data.

0. Some knowledge of BASh, Python and Fortran is required. You must know how to use UKCA, the UM and Monsoon3. If not, contact me or Luke and do this course first:
 
1. Either use my UM & UKCA branches:

or adapt your own UKCA code to output J-values.

2. Make output streams for them. Output the following STASH items at the desired time profiles:

And all of your J-values. If using my branches, they are:

I recommend using the T1H (aka T1HR) time profile for high-resolution data but any will work.

3. Run the UM for the desired length of time, outputting the specified STASH items. Make sure the STASH outputs are sent to MASS. For realistic conditions, I recommend using an SSP nudged suite with Fast-JX + lookup table at pressures less than 20 Pa, but you don't have to. If you want to copy my suite, it's at:
 
4. Fetch all of your relevant .pp output files from MASS and put them wherever you'll preprocess them.

5. Use the check_fields_pp.py script to see if it looks right.



Part B: How to preprocess UKCA output data to use as ML training inputs.

1. Do part A, above, or download my output data from MASS, which are:

2. Use and activate my conda environment, condaenv: 

conda activate condaenv
You need this exact environment because it contains a specific cfpython bug fix. If you use any other version of cfpython, the cf flatten function may not work and your program may get stuck. This cfpython version conflicts with matplotlib so you'll need to deactivate the conda environment before making any matplotlib plots:
conda deactivate

3. Use one of the flat_fields scripts to flatten the 5D UM output data to 2D arrays of features and samples, and convert to numpy files.
If you output your data on hourly timesteps or downloaded my data from MASS, use flat_fields_1982.py.
If you output your data on monthly timesteps, use flat_fields_long_run.py.
Adapt the script for your own file paths and any other changes required.
For any other output time profile, you may need to adapt one of these scripts.

4. You can use one of your .npy files generated in the above step if you want to skip this step, but for a good training dataset you should do this step.
Use the low_res_sample_train.py script. Adapt it for your own file paths and whatever other changes you need to make, ensuring that the output file it creates will fit in your computer's program memory. It makes a global dataset uniformly sampled from all of the .pp files.  

5. Use the check_fields_np.py script to see if it looks right.



Part C: How to repeat model comparisons experiments.
If you made your own UKCA code and output different J-values to me, or different input data, you'll need to change the indexing of the dataset in the Python scripts and make sure the right things are used as inputs and targets.

1. Do part B, above, or download and use my 1-year global data sample, at:

2. Use my conda environment, cenv, or your own Python environment with numpy and matplotlib:

conda activate cenv

3. Use the models_comparison.py script, adapting file paths and whatever else you want to change. This will build several ML models and test them on the dataset.



Part D: How to train a random forest photolysis emulator for use in UKCA.
If you made your own UKCA code and output different J-values to me, or different input data, you'll need to change the indexing of the dataset in the Python scripts and make sure the right things are used as inputs and targets.

1. Do part B, above, or download and use my 1-year global data sample, at:

2. Use my conda environment, cenv, or your own Python environment with numpy and matplotlib:

conda activate cenv

3. Use the forest_save.py script, adapting file paths and whatever else you want to change. This will train the random forest and save it as a pickle file.

4. If you want to use the random forest in Fortran, use the forest_structure_save.py script, adapting file paths and whatever else you want to change. This converts the pickle object of the trained random forest into a NetCDF file, which can be read in Fortran.

5. Test the converted model still works, using the forest_structure_build.py script. A negligible discrepancy is expected.



Part E: How to integrate the emulator into UKCA and run it online.

1. Do part D, above, or download and use my saved random forest model at:

2. Get and use my UKCA and UM branches, at:

Make sure your suite will run with version 13.9. You can copy my suite if required:

3. Place the NetCDF file somehwere you can read in to UKCA. Adapt this UKCA file to read in your NetCDF file: 

4. Select photolysis STASH items in your UM suite, or whatever else you want to test the effect of the emulator on.

5. Run the UM. Photolysis should now be emulated with the random forest.
