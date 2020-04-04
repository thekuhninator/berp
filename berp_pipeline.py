import click
from berp.viz import *
import berp.helper as helper
from berp.viz.tsne import plot_tsne
from berp.viz.boxplot import single_boxplot
from berp.report_generator import generate_report
import click

@click.command()
@click.option('--path', help="The path to the dataset you wish to run the BERP pipeline for. The folder specified by " +
                                "the path should contain a gene counts file ending in gene_counts.csv and a metadata file ending in metadata")
def cli(dataset_path):
    click.echo('Hello World!')
    #test_dataset_paths = helper.get_dataset_path('C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Data/combined_dataset_filtered')
    test_dataset_paths = helper.get_dataset_path(dataset_path)
    test_dataset = helper.get_dataset(test_dataset_paths)
    #test_dataset = datasets[0]
    #test_dataset = helper.get_dataset('C:/Users/Roman/Documents/Work/Depression\_and\_Immunology/Spring_Research/Data/combined_dataset_filtered/')
    gene_counts, metadata, dataset_name = test_dataset[0], test_dataset[1], test_dataset[2]
    print(dataset_name)

    tsne_path = plot_tsne(gene_counts, metadata, 25, 'scid_diagnosis', '', dataset_name)
    boxplot_path = single_boxplot(test_dataset)

    # what do we need to send report... the image URL, and the name of the dataset
    generate_report('Batch_Correction_Report.html', dataset_name, boxplot_path, tsne_path)

    #plot_tsne
    #plot_tsne_all(datasets)

