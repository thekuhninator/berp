from berp import correct

# okay let's test some stuff...

# let's give it a data directory

# and have it do stuff.

# define path and test variables
test_gene_counts_path = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Test_Data/test/test_gene_counts.csv'
test_metadata_path    = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Test_Data/test/test_metadata.csv'

# limma correction
correct.limma(gene_counts_path = test_gene_counts_path,
      metadata_path    = test_metadata_path,
      factor_of_interest= 'scid_diagnosis')

# mean center correction
correct.mean_center()

# z-score correction
correct.mean_center()

# ComBat correction
correct.mean_center()



# let's get all the gene counts and metadata

