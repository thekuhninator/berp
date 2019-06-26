import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab
from matplotlib.legend_handler import HandlerBase
import subprocess
import seaborn as sns
import numpy
import random
import math

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

def get_datasets():
    datasets = []
    # get all folders in a directory
    directory = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/England_Research/Data/'
    data_folders = [x[0] for x in os.walk(directory)][1:]

    print(data_folders)
    #print(data_folders)
    # go through each folder and get the files
    for data_folder in data_folders:
        #if ('mean_centered' not in data_folder):
        #    continue
        subDir = data_folder + '/'
        data_files = [f for f in os.listdir(subDir)]
        # get gene counts
        print(data_files)
        gene_counts = subDir + [fileName for fileName in data_files if 'gene_counts' in fileName][0]
        # get meta data
        metadata = subDir + [fileName for fileName in data_files if 'metadata' in fileName][0]

        datasets.append((gene_counts, metadata, data_folder.split('/')[-1]))
    return datasets;

def get_geneCounts(gene_counts_path ,annot=None):
    # get the input files
    input_counts = gene_counts_path#"C:\\Users\\Roman\\Documents\\Work\\Depression_and_Immunology\\Data\\filteredData.csv"

    # get gene counts
    gene_counts = pd.read_csv(input_counts, index_col=0, header=0).T
    gene_counts.index = gene_counts.index.map(int)  # index is messed up for some reason

    if(not (annot == None)):
        # select only rows that appear in the annot file
        gene_counts = gene_counts.ix[annot.index.values]
        # we need to drop NaNs that were inserted from annot
    gene_counts = gene_counts.dropna()
    return gene_counts

def get_metadata(metadata_path):
    input_annot = metadata_path#"C:\\Users\\Roman\\Documents\\Work\\Depression_and_Immunology\\Scripts\\Data\\meta_data_with_paths_all_patients_final.csv"
    # get meta_data
    annot = pd.read_csv(input_annot).set_index('record_id')
    #annot = annot[annot['missing'] == False]

    return annot


'''
 .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. |
| |   ______     | || |     ______   | || |      __      | |
| |  |_   __ \   | || |   .' ___  |  | || |     /  \     | |
| |    | |__) |  | || |  / .'   \_|  | || |    / /\ \    | |
| |    |  ___/   | || |  | |         | || |   / ____ \   | |
| |   _| |_      | || |  \ `.___.'\  | || | _/ /    \ \_ | |
| |  |_____|     | || |   `._____.'  | || ||____|  |____|| |
| |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------' 
'''

def plot_pca(gene_counts_path, metadata_path, folder_name,  plot_name = None):
    # run pca
    #Rscript
    #PCA_script.R - g / path / to / gene.counts - a / path / to / gene.annotation
    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/' + folder_name

    try:
        os.mkdir(output_dir)
    except:
        print('it already exists okay')

    #subprocess.call("Rscript C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/PCA/PCA_script.R -g "
    subprocess.call(
        "Rscript C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/PCA/PCA_script_png.R -g "
                    + gene_counts_path + ' -a ' + metadata_path + ' -o ' + output_dir + ' -f ' + folder_name, shell=True)

'''
 .----------------.  .----------------.  .----------------.  .-----------------. .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |  _________   | || |              | || |    _______   | || | ____  _____  | || |  _________   | |
| | |  _   _  |  | || |              | || |   /  ___  |  | || ||_   \|_   _| | || | |_   ___  |  | |
| | |_/ | | \_|  | || |    ______    | || |  |  (__ \_|  | || |  |   \ | |   | || |   | |_  \_|  | |
| |     | |      | || |   |______|   | || |   '.___`-.   | || |  | |\ \| |   | || |   |  _|  _   | |
| |    _| |_     | || |              | || |  |`\____) |  | || | _| |_\   |_  | || |  _| |___/ |  | |
| |   |_____|    | || |              | || |  |_______.'  | || ||_____|\____| | || | |_________|  | |
| |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 
'''

def plot_tsne(df,annot, perplexity, plot_name = None):

    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/' + plot_name + '/'
    model = TSNE(n_components=2, perplexity=perplexity, n_iter = 5000)
    model = model.fit_transform(df)

    list_color = ['k', 'k', 'k', 'r', 'g', 'b', 'y']
    list_marker = ["P", "d", "^", "o", "o", "o", "o"]
    list_lab = ['Healthy', 'Depression', 'Bipolar', 'Batch 1 (Toups)', 'Batch 2 (Toups)', 'Batch 3 (Toups)', 'Batch 4 (DGN Dataset)']

    tsne_df = pd.DataFrame()
    tsne_df['x'] = model[:, 0]
    tsne_df['y'] = model[:, 1]
    batchValues = annot['batch'].map({0.0: 'w', 1.0:'r', 2.0:'g', 3.0:'b', 4.0:'y'})

    x = [1, 2, 3]
    y = [4, 5, 6]


    ax = plt.gca()

    ax.set_facecolor('#ffffff')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('red')
   # for edge_i in ['top', 'bottom', 'right', 'left']:
    #    ax.spines[edge_i].set_edgecolor("#000000")


    diagnosisValues = annot['scid_diagnosis'].map({0.0: 'P', 1.0:'d', 2.0:'^', 3.0:'^', 4.0:'P'})

    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                           marker=tup[1],color=tup[0], transform=trans)]

    for x,y,b,d in zip(tsne_df['x'], tsne_df['y'], batchValues, diagnosisValues):
        ax.scatter(x, y, c=b, marker=d)

    ax.legend(list(zip(list_color, list_marker)), list_lab, handler_map={tuple:MarkerHandler()})

    print('WEREA BOUT TO SAVE THE T-SNE FIGURE')
    print(output_dir + 't-sne_perplex_' + str(perplexity) + '_' + plot_name + '.png')

    plot_title = ' '.join([x.capitalize() for x in plot_name.split('_')])

    plt.title(plot_title + ' - T-SNE Perplexity: ' + str(perplexity))
    plt.savefig(output_dir + 't-sne_perplex_' + str(perplexity) + '_' + plot_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

'''
 .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. |
| |  ___  ____   | || |   ______     | || |  _________   | || |  _________   | |
| | |_  ||_  _|  | || |  |_   _ \    | || | |_   ___  |  | || | |  _   _  |  | |
| |   | |_/ /    | || |    | |_) |   | || |   | |_  \_|  | || | |_/ | | \_|  | |
| |   |  __'.    | || |    |  __'.   | || |   |  _|  _   | || |     | |      | |
| |  _| |  \ \_  | || |   _| |__) |  | || |  _| |___/ |  | || |    _| |_     | |
| | |____||____| | || |  |_______/   | || | |_________|  | || |   |_____|    | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 
'''

def run_kBET(test_data):

    print(test_data[2])
#    if(test_data)
    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/' + test_data[2]
    try:
        os.mkdir(output_dir)
    except:
        print('it already exists okay')

        # except FileExistsError:
        # otherwise we're fine it already exists


    #print(x)
    command = 'Rscript'
    path_to_script = "C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/kBET/kBET_script_pipeline.R"

    # print('kbet THIRD ARGUMENT')
    # print(test_data[2])

    args = [test_data[0], test_data[1], output_dir, test_data[2]]

    # build command
    cmd = [command, path_to_script] + args

    # NEED TO UNCOMMNE THIS

    #x = subprocess.check_output(cmd, universal_newlines=True, shell=True)
    #print(x)

    # now let's get the mean rejection rate
    kBetResultsName = output_dir + '/' + test_data[2] + '_kBET_results.csv'
    # first let's get the csv
    kBet_results = pd.read_csv(kBetResultsName)
    rejection_rate = kBet_results.ix[0, 2]
    acceptance_rate = 1 - rejection_rate

    return acceptance_rate


def getMetrics(test_data):

    print(test_data[2])

    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/' + test_data[2]
    try:
        os.mkdir(output_dir)
    except:
        print('it already exists okay')
        # except FileExistsError:
        # otherwise we're fine it already exists

    command = 'Rscript'
    path_to_script = "C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/kBET/kBET_script_pipeline.R"

    #print('kbet THIRD ARGUMENT')
    #print(test_data[2])

    args = [test_data[0], test_data[1], output_dir, test_data[2]]

    # build command
    cmd = [command, path_to_script] + args

    #NEED TO UNCOMMNE THIS

    #x = subprocess.check_output(cmd, universal_newlines=True, shell=True)

    #print(x)


    # now let's get the mean rejection rate
    kBetResultsName = output_dir + '/' + test_data[2] + '_kBET_results.csv'
    # first let's get the csv
    kBet_results = pd.read_csv(kBetResultsName)

    rejection_rate = kBet_results.ix[0, 2]
    acceptance_rate = 1 - rejection_rate

    print(test_data[2])
    print('kBET acceptance ' + str(acceptance_rate))
    print('Avedist ' + str(kBet_results.ix[0,4]))
    print('Pvcam ' + str(kBet_results.ix[0,5]))
    print('skewdiv ' + str(kBet_results.ix[0,6]))
    print('kldist ' + str(kBet_results.ix[0,7]))
    print()

    rejection_rate = kBet_results.ix[0, 2]
    acceptance_rate = 1 - rejection_rate

    return acceptance_rate

'''
 .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. |
| |   ______     | || |     ____     | || |  ____  ____  | |
| |  |_   _ \    | || |   .'    `.   | || | |_  _||_  _| | |
| |    | |_) |   | || |  /  .--.  \  | || |   \ \  / /   | |
| |    |  __'.   | || |  | |    | |  | || |    > `' <    | |
| |   _| |__) |  | || |  \  `--'  /  | || |  _/ /'`\ \_  | |
| |  |_______/   | || |   `.____.'   | || | |____||____| | |
| |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------' 
'''

def boxPlot(datasets):

    # here's what we want to do
    # get the mean expression for each sample within a batch
    # each box plot should be a BATCH
    # group different types of correction together.

    sns.set(style="ticks", palette="pastel")
#    plt.set_xticklabels(rotation=45)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    #fig.canvas.set_window_title('TITLE THING')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    # let's build the dataset
    # let's get the list of

    boxPlotData = pd.DataFrame()
    for dataset in datasets:

        gene_counts = dataset[0]
        metadata = dataset[1]
        name = dataset[2]

        print('BOXPLOT ')
        print(name)

        # we need to iterate through each batch
        batches = np.unique(metadata['batch'])
        for batchNum in batches:
            # let's get the subset of the data
            relevent_metadata = metadata.loc[metadata['batch'] == batchNum]
            gene_counts_batch = gene_counts.ix[ relevent_metadata.index.values ]

            #print(relevent_metadata)
            #print(gene_counts_batch)

            meanValues = gene_counts_batch.mean(axis=0)

            batch = np.ones(len(meanValues), ) * batchNum
            newDf = pd.DataFrame({'Gene Counts': meanValues, 'Batch': batch})


            #print(meanValues)
            newDf['name'] = name
            #            newDf = newDf.rename(columns = {'0' : 'Gene Counts'})
            #print(newDf)

            #        newDf.columns[0] = 'Gene Counts'
            boxPlotData = boxPlotData.append(newDf)


    print('****** BOX PLOT DATA*******(')
    #print(boxPlotData)
    print('****** END BOX PLOT DATA*******(')

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)

    #import seaborn as sns
    planets = sns.load_dataset("planets")
    g = sns.factorplot("year", data=planets, aspect=1.5, kind="count", color="b")
    g.set_xticklabels(rotation=30)
    plt.show()

    ax = sns.boxplot(x="name", y="Gene Counts",
                hue="Batch", palette=["m", "g", "r", "b"],
                data=boxPlotData)

    plt.title("Comparative Boxplot Gene Averages", fontsize = 24)
    plt.xlabel('Dataset', fontsize = 18)
    plt.ylabel('Gene Counts', fontsize = 18)

    #plt.xticks(rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.setp(ax.get_xticklabels(), rotation=45)

    sns.despine(offset=10, trim=True)

    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/'

    plt.savefig(output_dir + 'comparative_boxplot' + '.png')

    plt.show()

        # x is correction method
    # hue is batch
    # y is gene counts
        # here's what we need
        # we need a list of means for each sample for each batch
        # so [mean of each sample in batch 1], mean of each sample in batch2

        # ticks, which correction each belongs to

        #


    # get the data


        #    boxplot_data = [ gene_data.mean() for data in gene_data]
'''
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('Gene data comparative box plot')
    fig.subplots_adjust(left=0.075, right=0.95, top=.9, bottom=.25)

    # add a horizontal line to the grid
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of Gene Counts from Different Batch Correction Methods', fontsize=20)
    ax1.set_xlabel('Gene Data', fontsize=18)
    ax1.set_ylabel('Gene Counts', fontsize=18)

    plt.boxplot(gene_data)

    numBoxes = len(gene_data)
    ax1.set_xlim(0.5, numBoxes + 0.5)
    ax1.set_xticklabels(dist_names, rotation=45, fontsize=8)
    #    plt.xticks([i for i in range(1,numBoxes+1)], dist_names)


    #    plt.boxplot(gene_data)
    plt.show()
    # Draw a nested boxplot to show bills by day and time
'''


def groupBoxPlot(datasets):
    # get the data
    gene_data = [ dataset[0].mean() for dataset in datasets]

    dist_names = [ dataset[2] for dataset in datasets]
    print(dist_names)



'''
 .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. |
| |  ____  ____  | || | ____   ____  | || |    ______    | |
| | |_   ||   _| | || ||_  _| |_  _| | || |  .' ___  |   | |
| |   | |__| |   | || |  \ \   / /   | || | / .'   \_|   | |
| |   |  __  |   | || |   \ \ / /    | || | | |    ____  | |
| |  _| |  | |_  | || |    \ ' /     | || | \ `.___]  _| | |
| | |____||____| | || |     \_/      | || |  `._____.'   | |
| |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------' 
'''
def makeTables(geneCountFile, annotFile):
    # read the annotation file
    annotTable = pd.read_csv(annotFile, sep=",", header=0, index_col=0)
    # read the gene counts file
    countTable = pd.read_csv(geneCountFile, sep=",", header=0, index_col=0).T
    countTable.index = countTable.index.map(int)  # index is messed up for some reason
    return (annotTable, countTable)


def sepBatches(countTable, annotTable, batchNum):
    # get meta_data
    # both_meta_data= pd.read_csv('C:/Users/Roman/Documents/Work/Depression_and_Immunology/Scripts/gender_analysis/meta_data/second_batch_mdd_both.csv').set_index('record_id')
    batchData = annotTable.loc[annotTable['batch'] == batchNum]
    #print(countTable.T.index.values)
    #print(countTable)
    return ((countTable).ix[batchData.index.values])

#    print(male_meta_data.index.values)

    # get both genders from second batch
    # both_gender_counts = gene_counts.ix[both_meta_data.index.values]
    # print(both_gender_counts.shape)
    # get only males

    #male_gender_counts = gene_counts.ix[male_meta_data.index.values]
'''
    #print('getting stuff')
    batchone = annotTable.loc[annotTable['batch'] == batchNum]
    #print('gotttemmmmm')
    ids = batchone.index.tolist()

    countTable.columns = countTable.columns.map(int)
    print(countTable)
    genebatch = countTable.iloc[:,countTable.columns.isin(ids)]
    print(countTable)
    return genebatch
'''

def topVarience(genebatch, percentage, name):
    #print(genebatch)
    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/HVG_Vars/' + name
    genebatch = genebatch.T
    genebatch['varience'] = genebatch.var(axis=1).tolist() # check the AXIS

    variSort = genebatch.sort_values('varience', ascending=False)
    topGenes = len(variSort.index) * percentage
    #print("My amount of gnees")
    #print(genebatch.shape[0] * percentage)
    #print('HER AMOUNT OF GENES')
    #print(topGenes)
    variSortTop = variSort.head(math.floor(topGenes))
    if('combined_dataset_unfiltered' in name and '1' in name):
        print('PRINTING ' + str(name))
        genebatch.to_csv(output_dir + 'ORIGINAL_GENES.csv')
        #print(genebatch)
        print()

    #print(variSortTop['varience'])
    #variSortTop['varience'].to_csv(output_dir + 'ORIGINAL_GENES.csv')

    #print(variSortTop)
    return variSortTop.index


def getHvgs(countTableF, annotTableF, name):
    #print('GETTING HVGS FOR NAME ' + name)

    # does the file already exist?
    fileName = name + '_hvgs.txt'
    hvgs = []
    # IF THE FILE EXISTS
    if( False ): #os.path.isfile(fileName)): *** CHANGE THIS
        with open(fileName) as f:
            content = f.readlines()
        hvgs = [line.strip() for line in content]
        print('THE NUMBER OF HVGS FOR ' + str(name) + ' is ' + str(len(hvgs)))
        return hvgs
    # IF IT DOESN'T EXIST
    else:
        annotTable, countTable = makeTables(countTableF, annotTableF)
        numBatches = np.amax(annotTable['batch'])
        #print("made it this far")
        batches = [sepBatches(countTable, annotTable, batchNum) for batchNum in range(1, numBatches + 1)]
        #print('*** TOP VAR FOR ' + name)
        batchVariances = [topVarience(batch, .1, name + str(i)) for i, batch in enumerate(batches) ]
        hvgs = batchVariances[0]

        for i in range(1, len(batchVariances)):
                # something might be wrong here idk??????????
            #   print(str(0) + " combined with  " + str(i))
            hvgs = hvgs.intersection(batchVariances[i])
        allInterList = hvgs.tolist()
        with open(name + '_hvgs.txt', 'w') as f:
            for item in allInterList:
                f.write(item)
                f.write("\n")
        return allInterList

    #print("the number of hvgs is " + str(len(allInterList)))


def getIntersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def getRetained(quantativeData):
    #print(quantativeData)
    beforeCorrectionHvgs = []
    newQuantativeData = []
    correction = ['z_score', 'combat', 'limma', 'pca_loading']

    for data in quantativeData:
        originalData = True
        name = data[0]
        hvgs = data[1]

        # get the original hvgs
        for c in correction:
            if(c in name):
                originalData = False

        if(originalData):
            beforeCorrectionHvgs.append((name, hvgs))

    # go through again and get the percent of HVGs retained

    print('Before correction hvgs')
    for iterObject in beforeCorrectionHvgs:
        print(iterObject[0])

    for data in quantativeData:
        name = data[0]
        hvgs = data[1]
        acceptanceRate = data[2]
        # iterate through the before correction hvgs
        for originalData in beforeCorrectionHvgs:
            originalDataName = originalData[0]
            originalDataHvgs = originalData[1]

            if(originalDataName in name):
                print('Oirignal data hvgs' + str(originalDataName))
                #print(originalDataHvgs)
                print('this HVGS name ' + str(name))
                #print(hvgs)
                divisor = len(originalDataHvgs)
                if(len(originalDataHvgs) == 0):
                    divisor = 1
                # then this is the dataset it belongs to and we should calcualte the hvg retained

                # should we now do a rank analysis????
                percentRetained = (len(getIntersection(originalDataHvgs, hvgs)) / divisor) * 100.0

                print('Name ' + name)
                print('before correction hvgs')
                print(beforeCorrectionHvgs)
                print("orignal HVGS " + str(len(originalDataHvgs)))
                print('new hvgs ' + str(len(hvgs)))
                print('intersection is ' + str(len(getIntersection(originalDataHvgs, hvgs))))
                print('divisor ' + str(len(originalDataHvgs)))
                print('Percent retained ' + str(percentRetained))
                print()

                newQuantativeData.append( (name, acceptanceRate, percentRetained) )
                #break

    return newQuantativeData

def kBet_plots(kBet_data):
#    x = [data[2] for data in kBet_data]
#    y = [data[1] for data in kBet_data]




    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                           marker=tup[1],color=tup[0], transform=trans)]

    for dataThing in kBet_data:
        print(dataThing[0])
        print(dataThing[1])
        print(dataThing[2])
        print()

    correction = ['z_score', 'combat', 'limma', 'mean_centered', 'pca_loading']
    dataset = ['toups', 'combined']
    filtration = ['unfiltered', 'filtered']
    colors = ['b','g','r','c','m', 'y']
    markers = [['*','d'],['s','P']]
#    markers = ['o','s','+','d','1', 'v','|','_']

    # create plot
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1, axisbg="1.0")


    # START NEW CODE

    fig, ax1 = plt.subplots(figsize=(10, 6))
    #fig.canvas.set_window_title('TITLE THING')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of IID Bootstrap Resampling Across Five Distributions')
    ax1.set_xlabel('Distribution')
    ax1.set_ylabel('Value')

    # Set the axes ranges and axes labels
 #   numBoxes = len(datasets)
 #   ax1.set_xlim(0.5, numBoxes + 0.5)
    top = 40
    bottom = -5

        # END NEW CODE
#print(kBet_data)



    plot_data = [(data[2], data[1], ) for i, data in enumerate(kBet_data)]


    for data in kBet_data:
        correctionMethod= ''
        datasetUsed = 0
        filterMethod = ''
        for i, c in enumerate(correction):
            print(c)
            if(c in data[0]):

                correctionMethod = colors[i]
                break
        if(correctionMethod == ''):
            print('mean_centered' in data[0])
            print('no color for ' + str(data[0]))
            correctionMethod = colors[4]
        for i, d in enumerate(dataset):
            if(d in data[0]):
                datasetUsed = i
                break
        for i, f in enumerate(filtration):
            if(f in data[0]):
                filterMethod = markers[datasetUsed][i]
                break
        #print(data)
        x = data[2]
        y = data[1]

        print('Correction method is ' + str(correctionMethod) + ' color is ' + data[0])
        #color = # this will be batch correction method
        #marker = # this will be filtered or unfiltered
        # we need for dataset
        ax1.scatter(x, y, c = correctionMethod, marker = filterMethod, s = 100, alpha=.5)

    correction = ['z_score', 'combat', 'limma', 'mean_centered', 'pca_loading']
    dataset = ['toups', 'combined']
    filtration = ['unfiltered', 'filtered']
    colors = ['b','g','r','c','m','y','k']
    markers = [['|','_'],['o','+']]


#    list_color =  ['b', 'g', 'r', 'c', 'm', 'k', 'k', 'k', 'k']
#    list_marker = ["x", "x", "x", "x", 'x', "|", "_", "o", "+"]
#    list_lab = ['Z_Score', 'ComBat', 'Limma', 'PCA_Loading', 'No Correction', 'Toups Unfiltered', 'Toups Filtered', 'Combined Unfiltered', 'Combined Filtered']

    list_color =  ['b', 'g', 'r', 'c']
    list_marker = ["x", "x", "x", 'x']
    list_lab = ['Z_Score', 'ComBat', 'Limma', 'Mean Centered']

    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                           marker=tup[1],color=tup[0], transform=trans)]

    list_color  = ["b", "g", "r", "c", "m", "k", "k", "k", "k"]
    list_mak    = ["o","o","o", "o", "o", "*", "d", "s", "P"]
    list_lab    = ['Z_score','Combat','Limma', "Mean Centered", "No Correction", "Toups Unfiltered", "Toups Filtered", "Combined Unfiltered", "Combined Filtered"]

    ax = plt.gca()
    ax.legend(list(zip(list_color,list_mak)), list_lab,
              handler_map={tuple:MarkerHandler()})
#    ax.legend(list(zip(list_color, list_marker)), list_lab, handler_map={tuple:MarkerHandler()})
    plt.title("Percent HVGs Retained vs kBET Acceptance Rate", fontsize = 24)
    plt.xlabel('Percent of Highly Variable Genes Retained', fontsize = 18)
    plt.ylabel('kBET (Acceptance Rate)', fontsize = 18)


    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/'

    plt.savefig(output_dir + 'kBET_Hvgs_vs_acceptance_rate' + '.png')

    plt.show()
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


#    ax1.set_ylim(bottom, top)
#    ax1.set_xticklabels(np.repeat(randomDists, 2),
#                        rotation=45, fontsize=8)

'''
 .----------------.  .----------------.  .----------------.  .-----------------.
| .--------------. || .--------------. || .--------------. || .--------------. |
| | ____    ____ | || |      __      | || |     _____    | || | ____  _____  | |
| ||_   \  /   _|| || |     /  \     | || |    |_   _|   | || ||_   \|_   _| | |
| |  |   \/   |  | || |    / /\ \    | || |      | |     | || |  |   \ | |   | |
| |  | |\  /| |  | || |   / ____ \   | || |      | |     | || |  | |\ \| |   | |
| | _| |_\/_| |_ | || | _/ /    \ \_ | || |     _| |_    | || | _| |_\   |_  | |
| ||_____||_____|| || ||____|  |____|| || |    |_____|   | || ||_____|\____| | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 
'''
datasets = get_datasets()

quantativeData = []
# run kBET
for dataset in datasets:
    quantData = (dataset[2], run_kBET(dataset), getHvgs(dataset[0], dataset[1], dataset[2]))
    quantativeData.append(quantData)

# get HVGs
hvgParam = [ (qData[0], qData[2] , qData[1]) for qData in quantativeData  ]
hvgsRetained = getRetained( hvgParam )
# kBET plots
kBet_plots(hvgsRetained)

for dataset in datasets:
    geneCountsPath = dataset[0]
    metadataPath = dataset[1]
    name = dataset[2]
    # plot pca
    plot_pca(geneCountsPath, metadataPath, name)

    # get metrics
    getMetrics(dataset)

    #plot_pca(geneCountsPath, metadataPath, name)
    datasetGeneCounts = get_geneCounts(dataset[0])
    datasetMetadata = get_metadata(dataset[1])
    datasetName = dataset[2]
    #print('finding genes of interest')
    #geneValues.append(getGenesOfInterest(datasetGeneCounts, datasetMetadata, datasetName))

    #print('plotting t-sne')
    for i in range(10,51,10):
        plot_tsne(datasetGeneCounts, datasetMetadata, i, datasetName)


# get metrics
#for dataset in datasets:


# run boxplot
boxPlot(genes_and_metadata)

# run t-sne



print('DONE GETTING ALL METRICS')
print('DONE GETTING ALL METRICS')
print('DONE GETTING ALL METRICS')
print('DONE GETTING ALL METRICS')
print('DONE GETTING ALL METRICS')
print('DONE GETTING ALL METRICS')
print('DONE GETTING ALL METRICS')
print('DONE GETTING ALL METRICS')


'''
geneValues = []
# PLOT T-SNE
for dataset in datasets:
    geneCountsPath = dataset[0]
    metadataPath = dataset[1]
    name = dataset[2]
    #plot_pca(geneCountsPath, metadataPath, name)
    #datasetGeneCounts = get_geneCounts(dataset[0])
    #datasetMetadata = get_metadata(dataset[1])
    #datasetName = dataset[2]
    #print('finding genes of interest')
    #geneValues.append(getGenesOfInterest(datasetGeneCounts, datasetMetadata, datasetName))

    #print('plotting t-sne')
    #for i in range(10,51,10):
    #    plot_tsne(datasetGeneCounts, datasetMetadata, i, datasetName)


geneArray = numpy.asArray(geneValues)
numpy.savetxt("lrrn3.csv", geneArray, delimiter=",")
'''

'''
#print('DONE WITH PCA')

#genes_and_metadata = [(get_geneCounts(dataset[0]), get_metadata(dataset[1]), dataset[2], None) for dataset in datasets ]


print('DONE')
# GET HVGS
#numHvgs = [( getHvgs(dataset[0], dataset[1], dataset[2]), dataset[2] ) for dataset in datasets[0:5]]
#print('HVGS NUMEBERS ARE ')
#print(numHvgs)



#run_kBET()



# let's get the HVGs

#run boxplots







'''
#print(x)
#subprocess.call ("C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/kBET/kBET_script_pipeline.R " + test_data[0] + test_data[1])
'''




'''
'''

#plot_tsne(gene_counts, 20)
#plot_tsne(gene_counts, 30)

'''
