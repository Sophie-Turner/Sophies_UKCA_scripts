Here are Sophie's Python scripts for the PhD work. Most are for processing UKCA data and working with machine learning models and their data. 
All of the scripts were written to run on Cambridge chemistry department's atmospheric servers, except for .ipynb files, which are notebooks that can be run in Colab on a browser, and scripts in the 'jasmin' directory, which are specifically to be run on the JASMIN supercomputer.
Contact Sophie Turner at st838@cam.ac.uk for more info.
If unavailable, contact Luke Abraham at nla27@cam.ac.uk.
The supplied conda evironment, cenv, can be used for all Python scripts except the flat_fields scripts.  

There are several things you can do to repeat my work:
A. Output the UKCA & UM data you need to make ML training data.
B. Preprocess UKCA output data to use as ML training inputs.
C. Repeat model comparisons experiments.
D. Train a random forest photolysis emulator for use in UKCA.
E. Integrate the emulator into UKCA & UM version 13.9.
You don't need to do all of them. I have some saved data you can use to avoid some of these parts if you wish. See the instructions below for whichever task you need to do. You need access to MASS for parts A, B and E. 



Part A: How to output the UKCA & UM data you need to make ML training data.

0. Some knowledge of BASh, Python and Fortran is required. You must know how to use UKCA, the UM and Monsoon3. If not, contact me or Luke and do this course first: 
https://www.ukca.ac.uk/wiki/index.php/UKCA_Chemistry_and_Aerosol_Tutorials_at_UMvn13.9
 
1. Either use my UM & UKCA branches:
https://code.metoffice.gov.uk/trac/um/browser/main/branches/dev/sophieturner/vn13.9_um_dev
https://code.metoffice.gov.uk/trac/ukca/browser/main/branches/dev/sophieturner/um13.9_ukca_dev
and turn off ML photolysis in UKCA src/control/photolysis/interface/photol_ctl_mod.F90
or adapt your own UKCA code to output photolysis rate coefficients (J-values).

3. Make output streams for them. Output the following STASH items at the desired time profiles:
m01s00i010	specific humidity
m01s00i266	cloud fraction
m01s01i140	sza
m01s01i217	upward shortwave flux
m01s01i218	downward shortwave flux
m01s00i408	pressure
m01s16i004	temperature
m01s50i219	O3 col
And all of your J-values. If using my branches, they are:
m01s50i228	O3
m01s50i229	NO2
m01s50i500	nitrates
m01s50i501	HCHOr
m01s50i502	HCHOm
m01s50i503	MeCOCHO
m01s50i504	prodCO
m01s50i505	prodOH
m01s50i506	O2
m01s50i507	Cl2O2
m01s50i508	NO3strat
m01s50i509	O1D
m01s50i511	OCS
m01s50i513	SO3
m01s50i514	MeONO2
m01s50i515	NALD
m01s50i516	ISON
m01s50i540	MeCHO-MeOO
m01s50i541	propanal
m01s50i555	NO3
m01s50i557	H2O
m01s50i559	HOBr
m01s50i560	HOCl
m01s50i561	HNO3
m01s50i562	HNO4
m01s50i563	H2O2
m01s50i564	MeOOH
m01s50i565	O2-O3P
m01s50i567	O3
m01s50i568	N2O
m01s50i569	MACR
m01s50i570	MACROOH
m01s50i571	MeCHO-CH4
m01s50i652	NO
m01s50i653	NO2 (from Strat-trop)
I recommend using the T1H (aka T1HR) time profile for high-resolution data but any should work.

4. Run the UM for the desired length of time, outputting the specified STASH items. Make sure the STASH outputs are sent to MASS. For realistic conditions, you can use an SSP nudged suite with Fast-JX + lookup table at pressures less than 20 Pa, but you don't have to. If you want to copy my suite, do:
rosie copy u-dv846
 
6. Fetch all of your relevant .pp output files from MASS and put them wherever you'll preprocess them.

7. Use the check_fields_pp.py script to see if the output looks right.



Part B: How to preprocess UKCA output data to use as ML training inputs.

1. Do part A, above.

2. Use and activate my conda environment, condaenv: 
conda env create -f condaenv.yml
conda activate condaenv
You need this environment because it contains an essential cfpython bug fix. If you use any other version of cfpython, the cf flatten function may not work and your program may get stuck. This cfpython version conflicts with matplotlib so you'll need to deactivate the conda environment before making any matplotlib plots.

3. Use one of the flat_fields scripts to flatten the 5D UM output .pp data to 2D arrays of features and samples, and convert to numpy files.
If you output your data on hourly timesteps or downloaded my data from MASS, use flat_fields_1982.py.
If you output your data on monthly timesteps, use flat_fields_long_run.py.
Adapt the script for your own file paths and any other changes required.
For other output time profiles, you may need to adapt one of these scripts.

4. You can use one of your .npy files generated in the above step if you want to skip this step, but for a good training dataset you should do this step.
Use the low_res_sample_train.py script. Adapt it for your own file paths and whatever other changes you need to make, ensuring that the output dataset it creates will fit in your computer's program memory. It makes a global dataset randomly sampled from all of the .pp files.  

5. Use the check_fields_np.py script to see if it looks right.



Part C: How to repeat model comparisons experiments.
If you made your own UKCA code and output different J-values to me, or different input data, you'll need to change the indexing of the dataset in the Python scripts and make sure the right things are used as inputs and targets.

1. Do part B, above, or use my 1-year global data sample, 1982_45m.npy, which is at https://zenodo.org/uploads/20307039
This file is 8.3 GB. Make sure you have enough program memory for it.

2. Use my conda environment, cenv, or your own Python environment with numpy and matplotlib:
conda env create -f cenv.yml
conda activate cenv

3. Use the models_comparison.py script, adapting file paths and whatever else you want to change. This will build several ML models and test them on the dataset.



Part D: How to train a random forest photolysis emulator for use in UKCA.
If you made your own UKCA code and output different J-values to me, or different input data, you'll need to change the indexing of the dataset in the Python scripts and make sure the right things are used as inputs and targets.

1. Do part B, above, or use my 1-year global data sample, 1982_45m.npy, which is at https://zenodo.org/uploads/20307039

2. Use my conda environment, cenv, or your own Python environment with numpy and matplotlib:
conda env create -f cenv.yml
conda activate cenv

3. Use the forest_save.py script, adapting file paths and whatever else you want to change. This will train the random forest and save it as a pickle file.

4. If you want to use the random forest in Fortran, use the forest_structure_save.py script, adapting file paths and whatever else you want to change. This converts the pickle object of the trained random forest into a NetCDF file, which can be read in Fortran.

5. Test the converted model still works, using the forest_structure_build.py script. A negligible discrepancy is expected.



Part E: How to integrate the emulator into UKCA and run it online.

1. Do part D, above, or use my saved random forest model, rf_structure.nc, which is at https://zenodo.org/uploads/20307039 
See rf_metadata.txt for more info.

2. Get and use my UKCA and UM branches:
https://code.metoffice.gov.uk/trac/um/browser/main/branches/dev/sophieturner/vn13.9_um_dev
https://code.metoffice.gov.uk/trac/ukca/browser/main/branches/dev/sophieturner/um13.9_ukca_dev 
Make sure your suite will run with version 13.9. You can copy my suite if required:
rosie copy u-dv846

4. Place the NetCDF file somewhere you can read in to UKCA. Adapt the 'filepath' variable on line 156 of this UKCA file to read in your NetCDF file: 
src/science/photolysis/ml/ml_photol_calc.F90

5. Select photolysis STASH items in your UM suite, or whatever else you want to test the effect of the emulator on.

6. Run the UM. Photolysis should now be emulated with the random forest.
