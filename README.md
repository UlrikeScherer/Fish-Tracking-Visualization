# Fish-Tracking-Visualization
Project: Developing Exploration Behavior

Holds the scripts to visualize the molly's trajectory. 
The resulting pdfs are contained in the folder [tex/trajectory](tex/trajectory).


#### Build 
To compile the *Cython* code run 
+ Mac: `CC='gcc-11' python3 setup.py build_ext --inplace`
+ Unix: `python3 setup.py build_ext --inplace`

#### Run 
It requires to have the Folders `FE_block1_autotracks_front` and `FE_block1_autotracks_back` in the parent directory containing the trajectory data. Accessing the data from the serve is clearly possible but very slow.  
To generate the trajectory visualizations 
`python3 main.py` and then `cd tex` to run the latex script:
`bash build-trajectories.sh`

#### Metrics: 
+ step length is the length of the vector drawn between to consecutive data frames. 
+ the mean and standard derivation illustrated in the visualization is computed from filtered data frames, removing obvious error point and normed by the distance between data frame when erroneous data point where removed. 
+ The number of spikes is defined by the threshold of $` > \mu + 3 \sigma`$
+ For the sum of angles we take each angle between consecutive steps anti-clockwise $`\alpha \in [-\pi, \pi]`$. 
+ For the average angle each angle $`\alpha > 0`$

#### Next Steps
For the metrics 
* average activity
* space used: Entropy heat map
* direction turning angle

Compute: `function(fish_id, time_interval in sec)`
+ compute SD/SE
+ store output of the function for some *time_interval* as **csv**.

+ Example of entropy heat map. 
+ activity over time plot

#### Open TODOs
+ pdfs for windows -- adapted the root to `C:\data\...`. (Not needed for now)
+ `\href` in windows not working. 
+ start BioTracker from link. Install BioTracker on Mac? 
Further documentation will follow here... 
### How to start the BioTracker
`.\BioTracker.lnk --video="X:\1_FE_(fingerprint_experiment)_SepDec2021\FE_recordings\FE_block1_recordings\FE_block1_recordings_week4\23442333\20211006_060000.23442333_front\23442333_20211006_060000.23442333_000000.mp4"`
