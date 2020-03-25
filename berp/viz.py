from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.legend_handler import HandlerBase
import pandas as pd
import numpy as np
import seaborn as sns
import subprocess
import os

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

'''
Plot PCA: Creates a PCA plot with the given gene_counts, metadata.
'''
def plot_pca(gene_counts, metadata, output_path):
    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(gene_counts)
    principalDf = pd.DataFrame(data= principalComponents,
                               columns=['PC1', 'PC2'])

    cdict = {0:' red', 1:'green'}
    label={0:'Label 1', 1: 'Label 2'}

    fig, ax = plt.subplots(figsize=(7,5))
    fig.patch.set_facecolor('white')

    for l in np.unique(labels):
        ix=np.where(labels == 1)
        ax.scatter()
    #finalDf = pd.concat([principalDf, df[['target']]], axis=1)

    # we can have them specify factors of interest when doing the stuff





def plot_pca(gene_counts_path, metadata_path, folder_name, output_dir,  plot_name = None):
    # run pca
    #Rscript
    #PCA_script.R - g / path / to / gene.counts - a / path / to / gene.annotation

    # we need to take the
    #output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/' + folder_name

    # TODO: we should clean up this try block
    # we should clean s
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

# TODO: we need to make this to work for whatever variables they pass in...
def plot_tsne(gene_counts ,metadata, perplexity, output_dir, plot_name = None):

    save_file = os.path.join(output_dir, plot_name)

    model = TSNE(n_components=2, perplexity=perplexity, n_iter = 5000)
    model = model.fit_transform(df)

    # TODO: we need to figure out how we want to do this with t-sne...
    list_color = ['k', 'k', 'k', 'r', 'g', 'b', 'y']
    list_marker = ["P", "d", "^", "o", "o", "o", "o"]
    list_lab = ['Healthy', 'Depression', 'Bipolar', 'Batch 1 (Toups)', 'Batch 2 (Toups)', 'Batch 3 (Toups)', 'Batch 4 (DGN Dataset)']

    tsne_df = pd.DataFrame()
    tsne_df['x'] = model[:, 0]
    tsne_df['y'] = model[:, 1]

    # TODO: this needs to take each as factors and then map them to colors
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

def plot_tsne_all(datasets):
    for dataset in datasets: # iterate through the dataests
        print(dataset)
        gene_counts = dataset[0]
        metadata = dataset[1]
        perplexity = 25
        #output_dir = dta
        #plot name is...

        #output_dir =
        #def plot_tsne(df, annot, perplexity, output_dir, plot_name=None):
        #plot_tsne


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

def plot_boxplot(datasets):

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
        print('WTF WHY ISNT THIS PRINTING')
        print(metadata)
        print(metadata['batch'])
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

    #plt.savefig(output_dir + 'comparative_boxplot' + '.png')

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