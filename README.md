# doPlan

Usage:
1. Install nuScenes devkit: https://www.nuscenes.org/nuplan
2. Download desired nuPlan data as described in the nuPlan documentation. Note that pointclouds are not necessary for these scripts to work.
3. Use videoGenWithSettings.py to generate videos for each scenario; specifications must be given in videoSettings.txt
4. Use userLabeler.py to run the labeling interface, specifications must be given in settings.txt
5. Access data in the csvs generated into outputs folder (or wherever the user specifies in settings.txt)