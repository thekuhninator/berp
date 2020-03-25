from .helper import get_geneCounts
from .helper import get_metadata
import os
import pandas as pd
import numpy as np
import subprocess

'''
 .----------------. 
| .--------------. |
| |   ________   | |
| |  |  __   _|  | |
| |  |_/  / /    | |
| |     .'.' _   | |
| |   _/ /__/ |  | |
| |  |________|  | |
| |              | |
| '--------------' |
 '----------------' 
'''


def z_correction(gene_counts_path, metadata_path):
    # let's get the files
    gene_counts = get_geneCounts(gene_counts_path)
    annotTable = get_metadata(metadata_path)

    data_folder = gene_counts_path.split('/')[-2] # uh is this allowed
    print('Data folder ' + data_folder)

    numBatches = np.amax(annotTable['batch'])
    correctedBatches = pd.DataFrame()

    for batchNum in range(1, numBatches + 1):
        print("Doing batch " + str(batchNum))
        # get all of the gene data related to this batch
        batchData = annotTable.loc[annotTable['batch'] == batchNum]
        batchCounts = ((gene_counts).ix[batchData.index.values])


        # calculate the z-scores
        batch_z_scores = (batchCounts - batchCounts.mean()) / batchCounts.std(ddof=1)
        #batch_z_scores = batch_z_scores[batch_z_scores.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]  # .astype(np.float64) ?
        batch_z_scores = batch_z_scores.fillna(0) # in the case that the standard deviation is zero it will divide by zero...
        correctedBatches = correctedBatches.append(batch_z_scores)


    # now we can save the dataframe off!
    fileNameLen = len(gene_counts_path.split('/')[-1])
    output_dir = gene_counts_path[:len(gene_counts_path) - fileNameLen] + data_folder + "_z_score"
    output_file = gene_counts_path.split('/')[-1].split('.')[0] + "_z_score_gene_counts.csv"

    print(output_dir)
    print(output_file)
    # if the output directory already exists
    if(not os.path.exists(output_dir)):
        try:
            os.mkdir(output_dir)
        except:
            print('Unable to create the output_dir {}'.format(output_dir))
    # save the z-scores off now
    correctedBatches.T.to_csv(output_dir + "/" + output_file)

'''
 .----------------.  .----------------.  .----------------.  .-----------------.
| .--------------. || .--------------. || .--------------. || .--------------. |
| | ____    ____ | || |  _________   | || |      __      | || | ____  _____  | |
| ||_   \  /   _|| || | |_   ___  |  | || |     /  \     | || ||_   \|_   _| | |
| |  |   \/   |  | || |   | |_  \_|  | || |    / /\ \    | || |  |   \ | |   | |
| |  | |\  /| |  | || |   |  _|  _   | || |   / ____ \   | || |  | |\ \| |   | |
| | _| |_\/_| |_ | || |  _| |___/ |  | || | _/ /    \ \_ | || | _| |_\   |_  | |
| ||_____||_____|| || | |_________|  | || ||____|  |____|| || ||_____|\____| | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 
'''


def mean_center(gene_counts_path, metadata_path):
    # let's get the files
    gene_counts = get_geneCounts(gene_counts_path)
    annotTable = get_metadata(metadata_path)

    data_folder = gene_counts_path.split('/')[-2] # uh is this allowed
    print('Data folder ' + data_folder)

    numBatches = np.amax(annotTable['batch'])
    correctedBatches = pd.DataFrame()

    for batchNum in range(1, numBatches + 1):
        print("Doing batch " + str(batchNum))
        # get all of the gene data related to this batch
        batchData = annotTable.loc[annotTable['batch'] == batchNum]
        batchCounts = ((gene_counts).ix[batchData.index.values])


        # mean center
        batch_mean_centered = (batchCounts - batchCounts.mean())
        correctedBatches = correctedBatches.append(batch_mean_centered)

    # now we can save the dataframe off!
    fileNameLen = len(gene_counts_path.split('/')[-1])
    output_dir = gene_counts_path[:len(gene_counts_path) - fileNameLen] + data_folder + "_z_score"
    output_file = gene_counts_path.split('/')[-1].split('.')[0] + "_z_score_gene_counts.csv"

    print(output_dir)
    print(output_file)
    # if the output directory already exists
    if(not os.path.exists(output_dir)):
        try:
            os.mkdir(output_dir)
        except:
            print('Unable to create the output_dir {}'.format(output_dir))
    # save the mean centered df off now
    correctedBatches.T.to_csv(output_dir + "/" + output_file)


'''
 .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |   _____      | || |     _____    | || | ____    ____ | || | ____    ____ | || |      __      | |
| |  |_   _|     | || |    |_   _|   | || ||_   \  /   _|| || ||_   \  /   _|| || |     /  \     | |
| |    | |       | || |      | |     | || |  |   \/   |  | || |  |   \/   |  | || |    / /\ \    | |
| |    | |   _   | || |      | |     | || |  | |\  /| |  | || |  | |\  /| |  | || |   / ____ \   | |
| |   _| |__/ |  | || |     _| |_    | || | _| |_\/_| |_ | || | _| |_\/_| |_ | || | _/ /    \ \_ | |
| |  |________|  | || |    |_____|   | || ||_____||_____|| || ||_____||_____|| || ||____|  |____|| |
| |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 
'''


def limma(gene_counts_path, metadata_path, factor_of_interest=None):

    # create the name for the output_dir
    data_folder = gene_counts_path.split('/')[-2] # uh is this allowed
    fileNameLen = len(gene_counts_path.split('/')[-1])

    #output_dir  = gene_counts_path[:len(gene_counts_path) - fileNameLen] + data_folder + "_z_score"
    output_dir = '/'.join(gene_counts_path.split('/')[0:-2]) + '/' + data_folder + '_limma'
    print(output_dir)
    output_file_gene_counts = output_dir + '/' + data_folder + '_limma_gene_counts.csv'
    output_file_metadata    = output_dir + '/' + data_folder + '_limma_metadata.csv'
    #print(output_file_gene_counts)
    #print('better output dir {}'.format(output_dir))
    # if the output directory already exists
    if(not os.path.exists(output_dir)):
        try:
            os.mkdir(output_dir)
        except:
            print('Unable to create the output_dir {}'.format(output_dir))

    command = 'Rscript'
    path_to_script = os.path.dirname(os.path.abspath(__file__)).replace('\\','/') + '/limma.R'
    print(path_to_script)
    #path_to_script = "limma.R"

    args = [gene_counts_path, metadata_path, output_dir, output_file_gene_counts, output_file_metadata]
    if(factor_of_interest != None):
        args.append(factor_of_interest)
    # build command
    cmd = [command, path_to_script] + args
    x = ''
    try:
        x = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    print(x)

    #'we should print the error if something goes wrong'


'''
 .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |     ______   | || |     ____     | || | ____    ____ | || |   ______     | || |      __      | || |  _________   | |
| |   .' ___  |  | || |   .'    `.   | || ||_   \  /   _|| || |  |_   _ \    | || |     /  \     | || | |  _   _  |  | |
| |  / .'   \_|  | || |  /  .--.  \  | || |  |   \/   |  | || |    | |_) |   | || |    / /\ \    | || | |_/ | | \_|  | |
| |  | |         | || |  | |    | |  | || |  | |\  /| |  | || |    |  __'.   | || |   / ____ \   | || |     | |      | |
| |  \ `.___.'\  | || |  \  `--'  /  | || | _| |_\/_| |_ | || |   _| |__) |  | || | _/ /    \ \_ | || |    _| |_     | |
| |   `._____.'  | || |   `.____.'   | || ||_____||_____|| || |  |_______/   | || ||____|  |____|| || |   |_____|    | |
| |              | || |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 
'''


def combat(gene_counts_path, metadata_path, factor_of_interest=None):

    # create the name for the output_dir
    data_folder = gene_counts_path.split('/')[-2] # uh is this allowed
    fileNameLen = len(gene_counts_path.split('/')[-1])

    #output_dir  = gene_counts_path[:len(gene_counts_path) - fileNameLen] + data_folder + "_z_score"
    output_dir = '/'.join(gene_counts_path.split('/')[0:-2]) + '/' + data_folder + '_limma'
    print(output_dir)
    output_file_gene_counts = output_dir + '/' + data_folder + '_limma_gene_counts.csv'
    output_file_metadata    = output_dir + '/' + data_folder + '_limma_metadata.csv'
    #print(output_file_gene_counts)
    #print('better output dir {}'.format(output_dir))
    # if the output directory already exists
    if(not os.path.exists(output_dir)):
        try:
            os.mkdir(output_dir)
        except:
            print('Unable to create the output_dir {}'.format(output_dir))

    command = 'Rscript'
    path_to_script = os.path.dirname(os.path.abspath(__file__)).replace('\\','/') + '/limma.R'
    print(path_to_script)
    #path_to_script = "limma.R"

    args = [gene_counts_path, metadata_path, output_dir, output_file_gene_counts, output_file_metadata]
    if(factor_of_interest != None):
        args.append(factor_of_interest)
    # build command
    cmd = [command, path_to_script] + args
    x = ''
    try:
        x = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    print(x)

    #'we should print the error if something goes wrong'
