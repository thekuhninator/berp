import os
import pandas as pd

'''
 .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. |
| |  ________    | || |      __      | || |  _________   | || |      __      | |
| | |_   ___ `.  | || |     /  \     | || | |  _   _  |  | || |     /  \     | |
| |   | |   `. \ | || |    / /\ \    | || | |_/ | | \_|  | || |    / /\ \    | |
| |   | |    | | | || |   / ____ \   | || |     | |      | || |   / ____ \   | |
| |  _| |___.' / | || | _/ /    \ \_ | || |    _| |_     | || | _/ /    \ \_ | |
| | |________.'  | || ||____|  |____|| || |   |_____|    | || ||____|  |____|| |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 
'''
GENE_COUNTS = 0
METADATA = 1
NAME = 2

'''
This function takes as input the folder to the data directory (i.e. the directory in which all the folders for the data
are present) and as output outputs a list of tuples containing (gene counts data frame, metadata dataframe, and name of
dataset).
'''
def get_datasets(data_directory):
    dataset_paths = get_datasets_path(data_directory) # get the dataset paths
    datasets = []

    for dataset_path in dataset_paths: # iterate through the paths
        datasets.append( get_dataset(dataset_path) ) # put it in a tuple

    return datasets

'''
Given a dataset path, return the gene counts, metadata and dataset name.
'''
def get_dataset(dataset_path):
    gene_counts = get_geneCounts(dataset_path[GENE_COUNTS])  # get the gene counts
    metadata = get_metadata(dataset_path[METADATA])  # get the metadata
    dataset_name = dataset_path[NAME]  # get the names
    return (gene_counts, metadata, dataset_name)  # put it in a tuple

'''
This function takes as input the folder to the data directory (i.e. the directory in which all the folders for the data
are present.
'''
def get_datasets_path(data_directory):
    datasets = []

    # get all folders in a directory
    data_folders = [folder[0] for folder in os.walk(data_directory) if os.path.isdir(folder[0])][1:]
    # go through each folder and get the files
    for data_folder in data_folders:
        data_folder = os.path.normpath(data_folder) # this might not work if there are spaces...
        data_files = [f for f in os.listdir(data_folder)] # get the files in the directory

        # get gene counts
        gene_counts = os.path.join(data_folder, [fileName for fileName in data_files if 'gene_counts' in fileName][0])
        # get metadata
        metadata = os.path.join(data_folder, [fileName for fileName in data_files if 'metadata' in fileName][0])

        dataset_name = os.path.basename(os.path.normpath(data_folder)) # get dataest name
        dataset_path = data_folder # get dataset path

        datasets.append((gene_counts, metadata, dataset_name, dataset_path)) # crate tuple

    return datasets # return datasets


'''
This function takes as input the path to the gene counts file.
It returns a pandas dataframe.
'''
def get_geneCounts(gene_counts_path ,annot=None):
    # get the input files
    #input_counts = gene_counts_path#"C:\\Users\\Roman\\Documents\\Work\\Depression_and_Immunology\\Data\\filteredData.csv"

    # get gene counts
    gene_counts = pd.read_csv(gene_counts_path, index_col=0, header=0).T
    gene_counts.index = gene_counts.index.map(int)  # index is messed up for some reason

    if(not (annot == None)):
        # select only rows that appear in the annot file
        gene_counts = gene_counts.ix[annot.index.values]

    # we need to drop NaNs that were inserted from annot
    gene_counts = gene_counts.dropna()
    return gene_counts


'''
This function takes the metadata path as input and returns a pandas dataframe of the annotation file.
'''
def get_metadata(metadata_path):
    # get metadata
    annot = pd.read_csv(metadata_path, index_col=0)
    return annot


'''
 .----------------. .----------------. .-----------------..----------------. .----------------. 
| .--------------. | .--------------. | .--------------. | .--------------. | .--------------. |
| |    ______    | | |  _________   | | | ____  _____  | | |  _________   | | |    _______   | |
| |  .' ___  |   | | | |_   ___  |  | | ||_   \|_   _| | | | |_   ___  |  | | |   /  ___  |  | |
| | / .'   \_|   | | |   | |_  \_|  | | |  |   \ | |   | | |   | |_  \_|  | | |  |  (__ \_|  | |
| | | |    ____  | | |   |  _|  _   | | |  | |\ \| |   | | |   |  _|  _   | | |   '.___`-.   | |
| | \ `.___]  _| | | |  _| |___/ |  | | | _| |_\   |_  | | |  _| |___/ |  | | |  |`\____) |  | |
| |  `._____.'   | | | |_________|  | | ||_____|\____| | | | |_________|  | | |  |_______.'  | |
| |              | | |              | | |              | | |              | | |              | |
| '--------------' | '--------------' | '--------------' | '--------------' | '--------------' |
 '----------------' '----------------' '----------------' '----------------' '----------------' 
'''
def getGenesOfInterest(gene_counts, metadata, name):
    geneValue = gene_counts.loc[:,'LRRN3'].mean()
    print(name)
    print('LRRN3: ', geneValue)
    return geneValue

