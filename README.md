# Fish-Tracking-Visualization
Project: Developing Exploration Behavior

Holds the scripts to visualize the molly's trajectory. 
The resulting pdfs are contained in the folder [tex/trajectory](tex/trajectory).

#### Build 
To compile the cython code run 
+ Mac: `CC='gcc-11' python3 setup.py build_ext --inplace`
+ Unix: `python3 setup.py build_ext --inplace`

#### Run 
To generate the the trajectory visualizations 
`python3 main.py` and then `cd tex` to run the latex script:
`bash build-trajectories.sh`

Documentation will follow here... 