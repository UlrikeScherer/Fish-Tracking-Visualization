# Fish-Tracking-Visualization
Project: Developing Exploration Behavior

Holds the scripts to visualize the molly's trajectory. 
The folder [tex/trajectory](tex/trajectory) contains the resulting pdfs.
To make use of the links to the mp4 and csv-files -- connect to the server `loopbio_data`. Currently the links work for MacOX systems and best with [Adobe Reader](https://get.adobe.com/de/reader/). The Preview.app somehow has permission problems to open files, at least on my system. 


#### Build 
To compile the *Cython* code run 
+ Mac: `CC='gcc-11' python3 setup.py build_ext --inplace`
+ Unix: `python3 setup.py build_ext --inplace`

#### Trajectory Visualization PDFs
[tex/env.sh](tex/env.sh) contains the paths to the trajectory data. One can configure these to point to the correct location of the data. Reading the data directly from the server `loopbio_data` results in long running times. It is recommended use a external hard drive holding the data.  
Accessing the data from the server is very slow.  
To generate the trajectory visualizations, run 
`python3 main.py` and then `cd tex` to run the `latex` script:
`bash build-trajectories.sh`

#### Activity metrics
* run: `python3 main.py metric={metric}`
For the metric argument use one out of `metric={activity, angle, tortuosity, entropy}` 

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
* Tortuosity in 2-D

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


| Fish ID | Camera ID and Position |
|---|---|
| 0 | 23520289_front |
| 1 | 23520289_back |
| 2 | 23484201_front |
| 3 | 23484201_back |
| 4 | 23520258_front |
| 5 | 23520258_back |
| 6 | 23442333_front |
| 7 | 23442333_back |
| 8 | 23520268_front |
| 9 | 23520268_back |
| 10 | 23520257_front |
| 11 | 23520257_back |
| 12 | 23520266_front |
| 13 | 23520266_back |
| 14 | 23484204_front |
| 15 | 23484204_back |
| 16 | 23520278_front |
| 17 | 23520278_back |
| 18 | 23520276_front |
| 19 | 23520276_back |
| 20 | 23520270_front |
| 21 | 23520270_back |
| 22 | 23520264_front |
| 23 | 23520264_back |
