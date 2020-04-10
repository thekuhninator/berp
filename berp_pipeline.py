import click
from berp.viz import *
import berp.helper as helper
from berp.viz.tsne import plot_tsne
from berp.viz.boxplot import single_boxplot
from berp.report_generator import generate_report
from berp.kbet_py import fart_ass
from berp.metrics import kbet
import click

@click.command()
@click.option('--dataset_path', help="The path to the dataset you wish to run the BERP pipeline for. The folder specified by " +
                                "the path should contain a gene counts file ending in gene_counts.csv and a metadata file ending in metadata")
@click.option('--output_path', default='./', help="The path where the user would like the report and images to be outputted to.")
def cli(dataset_path, output_path):
    click.echo('Hello World!')
    click.echo('The inputed path')
    # make a note that no backslashes are allowed
    click.echo(str(dataset_path))
    #test_dataset_paths = helper.get_dataset_path('C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Data/combined_dataset_filtered')
    test_dataset_paths = helper.get_dataset_path(dataset_path)

    #test_dataset = datasets[0]
    #test_dataset = helper.get_dataset('C:/Users/Roman/Documents/Work/Depression\_and\_Immunology/Spring_Research/Data/combined_dataset_filtered/')
    print(test_dataset_paths)
    test_dataset = helper.get_dataset(test_dataset_paths)

    gene_counts, metadata, dataset_name = test_dataset[0], test_dataset[1], test_dataset[2]
    print(dataset_name)

    fart_ass(gene_counts, metadata, '.', 'kBET_test_asshole')
    return




    #tsne_path = plot_tsne(gene_counts, metadata, 25, 'scid_diagnosis', '', dataset_name)
    #boxplot_path = single_boxplot(test_dataset)

    # here we need to run kBET as well as the other metrics.
    gene_counts_path, metadata_path, dataset_name, dataset_root_path = test_dataset_paths
    kbet_path = kbet(gene_counts_path, metadata_path, '.', 'kbet_output_roman')

    # what do we need to send report... the image URL, and the name of the dataset
    #generate_report('Batch_Correction_Report.html', dataset_name, boxplot_path, tsne_path)

    #plot_tsne
    #plot_tsne_all(datasets)

