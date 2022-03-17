# Fish-Tracking-Visualization
Project: Developing Exploration Behavior

Holds the scripts to visualize the molly's trajectory. 
The folder [tex/trajectory](tex/trajectory) contains the resulting pdfs.
To make use of the links to the mp4 and csv-files -- connect to the server `loopbio_data`. Currently the links work for MacOX systems and best with [Adobe Reader](https://get.adobe.com/de/reader/). The Preview.app somehow has permission problems to open files, at least on my system. 

#### Requirements
+ python3, gcc-11
+ install dependencies `pip3 install -r requirements.txt `

#### Build 
To compile the *Cython* code run 
+`python3 setup.py build_ext --inplace` or `CC='gcc-11' setup.py build_ext --inplace`

#### Trajectory Visualization PDFs
[tex/env.sh](tex/env.sh) contains the paths to the trajectory data. One can configure these to point to the correct location of the data. Reading the data directly from the server `loopbio_data` results in long running times. It is recommended use a external hard drive holding the data.  
Accessing the data from the server is very slow.  
To generate the trajectory visualizations, run 
`python3 main.py program={trajectory, feeding}` and then `cd tex` to run the `latex` script:
`bash scripts/build-trajectories.sh` and `bash scripts/build-trajectories.sh --feeding` for the feeding trajectories. 

#### Activity metrics
* run: `python3 main.py program={metric} <<optional>> time_interval=<<default>>100`
For the metric argument use one out of `{activity, angle, tortuosity, entropy}`.

* run: `bash scripts/build_analytics.sh` to generate the pdfs. 

#### Metrics: 
+ step length is the length of the vector drawn between to consecutive data frames. 
+ the mean and standard derivation illustrated in the visualization is computed from filtered data frames, removing obvious error point and normed by the distance between data frame when erroneous data point where removed. 
+ The number of spikes is defined by the threshold of $` > \mu + 3 \sigma`$
+ For the sum of angles we take each angle between consecutive steps anti-clockwise $`\alpha \in [-\pi, \pi]`$. 
+ For the average angle each angle $`\alpha > 0`$


Compute: `function(fish_id, time_interval in sec)`
+ compute SD/SE
+ store output of the function for some *time_interval* as **csv**.

#### Open TODOs
+ pdfs for windows -- adapted the root to `C:\data\...`. (Not needed for now)
+ `\href` in windows not working. 
+ start BioTracker from link. Install BioTracker on Mac? 
Further documentation will follow here... 
+ How to start the BioTracker
`.\BioTracker.lnk --video="X:\1_FE_(fingerprint_experiment)_SepDec2021\FE_recordings\FE_block1_recordings\FE_block1_recordings_week4\23442333\20211006_060000.23442333_front\23442333_20211006_060000.23442333_000000.mp4"`


| Index | Camera ID | Position | ID |
|---|---|---|---|
| 0 | 23442333 | front | m1_01|
| 1 | 23442333 | back | m3_01|
|2 | 23484201 | front | m1_21|
|3 | 23484201 | back | m1_22|
|4 | 23484204 | front | m1_03|
|5 | 23484204 | back | m1_04|
|6 | 23520257 | front | m2_02|
|7 | 23520257 | back | m1_12|
|8 | 23520258 | front | m2_03|
|9 | 23520258 | back | m1_16|
|10 | 23520264 | front | m3_02|
|11 | 23520264 | back | m2_04|
|12 | 23520266 | front | m1_13|
|13 | 23520266 | back | m1_14|
|14 | 23520268 | front | m1_07|
|15 | 23520268 | back | m1_08|
|16 | 23520270 | front | m1_19|
|17 | 23520270 | back | m1_20|
|18 | 23520276 | front | m1_23|
|19 | 23520276 | back | m1_24|
|20 | 23520278 | front | m1_09|
|21 | 23520278 | back | m1_10|
|22 | 23520289 | front | m1_05|
|23 | 23520289 | back | m2_01|
