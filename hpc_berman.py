import motionmapperpy as mmpy
from src.data_factory.processing import *
from src.utils import *
from src.config import BLOCK, ROOT_LOCAL
from src.utils.tank_area_config import *
import h5py, hdf5storage, pickle, glob
import time

def factory_main():
    projectRoot = 'content'
    projectPath = '%s/Fish_moves_stw3'%projectRoot
    
    parameters = mmpy.setRunParameters()
    parameters.pcaModes = 3
    parameters.samplingFreq = 5
    parameters.maxF = 2.5
    parameters.minF = 0.01
    parameters.omega0 = 5
    parameters.projectPath = projectPath
    parameters.numProcessors = 16
    parameters.method="UMAP"
    parameters.useGPU=0
    parameters.training_numPoints = 5000    #% Number of points in mini-trainings.
    parameters.trainingSetSize = 10000  #% Total number of training set points to find. 
                                 #% Increase or decrease based on
                                 #% available RAM. For reference, 36k is a 
                                 #% good number with 64GB RAM.
    parameters.embedding_batchSize = 30000  #% Lower this if you get a memory error when 
                                            #% re-embedding points on a learned map.
    parameters.kmeans = 10
    os.makedirs(parameters.projectPath,exist_ok=True)
    mmpy.createProjectDirectory(parameters.projectPath)
    fish_keys = get_camera_pos_keys()

    compute_all_projections(parameters.projectPath,fish_keys, recompute=False)
    #normalize 
    parameters.normalize_func = return_normalization_func(parameters)
    print("Subsample from projections")
    mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
    print("Fit data / find embeddings")
    fit_data(parameters)
    print("Find Watershed...")
    startsigma = 4.2 if parameters.method == 'TSNE' else 1.0
    mmpy.findWatershedRegions(parameters, minimum_regions=parameters.kmeans, startsigma=startsigma, pThreshold=[0.33, 0.67],
                         saveplot=True, endident = '*_pcaModes.mat')
    print("Done!")
    
def fit_data(parameters):
    #tsne takes 19 mins
    tall = time.time()
    tfolder = parameters.projectPath+'/%s/'%parameters.method

    # Loading training data
    with h5py.File(tfolder + 'training_data.mat', 'r') as hfile:
        trainingSetData = hfile['trainingSetData'][:].T
    trainingSetData[trainingSetData==0] = 1e-12 # replace 0 with 1e-12
    # Loading training embedding
    with h5py.File(tfolder+ 'training_embedding.mat', 'r') as hfile:
        trainingEmbedding= hfile['trainingEmbedding'][:].T

    if parameters.method == 'TSNE':
        zValstr = 'zVals' 
    else:
        zValstr = 'uVals'

    projectionFiles = glob.glob(parameters.projectPath+'/Projections/*pcaModes.mat')
    for i in range(len(projectionFiles)):
        print('Finding Embeddings')
        t1 = time.time()
        print('%i/%i : %s'%(i+1,len(projectionFiles), projectionFiles[i]))


        # Skip if embeddings already found.
        if os.path.exists(projectionFiles[i][:-4] +'_%s.mat'%(zValstr)):
            print('Already done. Skipping.\n')
            continue

        # load projections for a dataset
        projections = hdf5storage.loadmat(projectionFiles[i])['projections']
        print(projections.shape, trainingSetData.shape)
        # Find Embeddings
        zValues, outputStatistics, clusters = mmpy.findEmbeddings(projections,trainingSetData,trainingEmbedding,parameters)

        # Save embeddings
        hdf5storage.write(data = {'zValues':zValues, 'clusters':clusters}, path = '/', truncate_existing = True,
                        filename = projectionFiles[i][:-4]+'_%s.mat'%(zValstr), store_python_metadata = False,
                          matlab_compatible = True)

        # Save output statistics
        with open(projectionFiles[i][:-4] + '_%s_outputStatistics.pkl'%(zValstr), 'wb') as hfile:
            pickle.dump(outputStatistics, hfile)

        del zValues,projections,outputStatistics
        
    print('All Embeddings Saved in %i seconds!'%(time.time()-tall))

if __name__ == "__main__":
    tstart = time.time()
    factory_main()
    tend = time.time()
    print("Running time:", tend - tstart, "sec.")
    
