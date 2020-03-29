from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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
def single_boxplot(dataset):

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

    gene_counts = dataset[0]
    metadata = dataset[1]
    name = dataset[2]

    print('BOXPLOT ')
    print(name)

    batches = np.unique(metadata['batch'])
    for batchNum in batches:
        # let's get the subset of the data
        relevent_metadata = metadata.loc[metadata['batch'] == batchNum]
        gene_counts_batch = gene_counts.ix[ relevent_metadata.index.values ]

        meanValues = gene_counts_batch.mean(axis=0)

        batch = np.ones(len(meanValues), ) * batchNum
        newDf = pd.DataFrame({'Gene Counts': meanValues, 'Batch': batch})

        newDf['name'] = name
        #newDf = newDf.rename(columns = {'0' : 'Gene Counts'})
        #print(newDf)

        #        newDf.columns[0] = 'Gene Counts'
        boxPlotData = boxPlotData.append(newDf)


    print('****** BOX PLOT DATA*******(')
    #print(boxPlotData)
    print('****** END BOX PLOT DATA*******(')

    '''
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)

    #import seaborn as sns
    planets = sns.load_dataset("planets")
    g = sns.factorplot("year", data=planets, aspect=1.5, kind="count", color="b")
    g.set_xticklabels(rotation=30)
    plt.show()
    '''
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

    # create output path and save the file
    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/'
    output_path = output_dir + 'comparative_boxplot.png'
    plt.savefig(output_path)

    #plt.show()
    return output_path
