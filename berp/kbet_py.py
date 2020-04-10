import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

def fart_ass(gene_counts, metadata, output_dir, output_name):

    # imports
    utils = importr('utils')
    utils.chooseBioCmirror(ind=1) # select the first mirror in the list

    if(not rpackages.isinstalled('kBET')):
        print('kBET not installed, installing kBET')
        utils.install_packages('../source_packages/kBET-master.zip', type="source")
    else:
        print('kBET alraedy installed')
    if (not rpackages.isinstalled('devtools')):
        print('devtools not installed, installing devtools')
        utils.install_packages('devtools')
    else:
        print('devtools alraedy installed')

    #devtools = importr('devtools')
    #devtools.install_github('theislab/kBET')


    kBET = importr('kBET')



    batch = metadata['batch']

    #
    ###
    #
    #
    #r_gene_counts = ro.conversion.py2rpy(gene_counts)
    results = kBET(gene_counts, batch, plot=False)

    #utils.install_packages('gutenbergr', repos='https://cloud.r-project.org')

    print('we made it this far')
    #packages = ['devtools', 'kBET', 'bapred', 'ggplot2']
    #packages_to_install = [p for p in packages if not rpackages.isinstalled(p)]


    #for package in packages_to_install:
    #    utils.install_packages(package, repos='https://cloud.r-project.org'

    # load imports
    #devtools = importr('devtools')
    #kBet = importr('kBET'))


    rstring = """
        library(devtools)
        library(kBET)
        library(bapred)
        library(ggplot2)
        
        
        avedistVal = avedist(new_data, as.factor(batch) )#, as.factor(diagnosis))
        pvcamVal = pvcam(new_data, as.factor(batch), as.factor(diagnosis))
        skewdivVal = skewdiv(new_data, as.factor(batch))
        #sepscoreVal = sepscore(new_data, as.factor(batch) )
        kldistVal = kldist(new_data, as.factor(batch))
        batch.estimate <- kBET(my_data, batch, plot = FALSE)
        
    """