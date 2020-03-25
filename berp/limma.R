# if limma isn't installed install it
list.of.packages <- c("limma")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(limma)

# get arguments from the command line
args = commandArgs(trailingOnly=TRUE)

# Collect arguments from command line
if(length(args) >= 5)
{
  input_counts = args[1]
  input_annot  = args[2]
  output_dir  = args[3]
  output_gene_file = args[4]
  output_metadata_file = args[5]
  factor_of_interest = NULL
} else if (length(args) == 6)
{
  factor_of_interest = args[6]
} else
{
  stop("Incorrect number of arguments supplied to Limma script.")
}

print(input_counts)
print(input_annot)
print(output_dir)
print(output_gene_file)
print(factor_of_interest)

# thing to look at
gene_counts_df = read.csv(file= input_counts, header = TRUE, row.names = 1, check.names = FALSE)
metadata_df =    read.csv(file= input_annot,  header = TRUE, row.names = 1, check.names = FALSE)

# make batch a factor
batchCF <- factor(metadata_df$batch) # taking out the levels argument

if(!is.null(factor_of_interest))
{
  # we can take this in as an arugment
  factor_of_interest_CF = unique(metadata_df[factor_of_interest])
  CF <- factor(metadata_df[factor_of_interest],levels=c(factor_of_interest_CF))
  designCF <- model.matrix(~CF)
  # remove batch
  correctedExpDataCF = removeBatchEffect(gene_counts_df, batch = batchCF, design = designCF)
} else
{
  correctedExpDataCF = removeBatchEffect(gene_counts_df, batch= batchCF)
}
# save the dataframe now
write.csv(correctedExpDataCF, file = output_gene_file)
write.csv(metadata_df, file = output_metadata_file)
