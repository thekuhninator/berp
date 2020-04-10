from berp.viz import *
import berp.helper as helper
from berp.viz.tsne import plot_tsne
from berp.viz.boxplot import single_boxplot
from berp.report_generator import generate_report
from berp.metrics import kbet
import click
#limma(gene_counts_path = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Test_Data/test/test_gene_counts.csv',
#      metadata_path    = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Test_Data/test/test_metadata.csv')

print('done')

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

def cli():
    print('hello world')
    click.echo("hello world!")

@click.command()
@click.option('--path', help="The path to the dataset you wish to run the BERP pipeline for. The folder specified by " +
                                "the path should contain a gene counts file ending in gene_counts.csv and a metadata file ending in metadata")
def berp(dataset_path):
    click.echo('Hello World!')
    #test_dataset_paths = helper.get_dataset_path('C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Data/combined_dataset_filtered')
    test_dataset_paths = helper.get_dataset_path(dataset_path)
    test_dataset = helper.get_dataset(test_dataset_paths)
    #test_dataset = datasets[0]
    #test_dataset = helper.get_dataset('C:/Users/Roman/Documents/Work/Depression\_and\_Immunology/Spring_Research/Data/combined_dataset_filtered/')
    gene_counts, metadata, dataset_name = test_dataset[0], test_dataset[1], test_dataset[2]
    print(dataset_name)

    #tsne_path = plot_tsne(gene_counts, metadata, 25, 'scid_diagnosis', '', dataset_name)
    #boxplot_path = single_boxplot(test_dataset)
    kbet_path = kbet(gene_counts, metadata, '.', 'kbet_output_roman')

    # what do we need to send report... the image URL, and the name of the dataset
    #generate_report('Batch_Correction_Report.html', dataset_name, boxplot_path, tsne_path)

    #plot_tsne
    #plot_tsne_all(datasets)






'''

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

'''

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


'''

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
