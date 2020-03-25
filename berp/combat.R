# ComBat script, path to gene_counts_df, metadata_df, boolean value for if using idsBin
# boolean value for if dataCorrection will be done, boolean value for if writing
# to file and the name of the file you want the results to be written to.
# Will return p value dataframe that compares categories to each other. 

library(sva)

# TRUE, FALSE, TRUE, "toups_filtered_combat_gene_counts.csv"

combatScript <- function(expPath, metadata_dfPath, idsBin, dataCorrection, write, writeName) {
  idsBin = TRUE
  dataCorrection = FALSE
  write = TRUE
  # set up initial data, etc
  gene_counts_df = read.csv(expPath, check.names = FALSE)
  metadata_df = read.csv(metadata_dfPath, check.names = FALSE)
  
  rownames(metadata_df) = metadata_df[,1]
  rownames(gene_counts_df) <- gene_counts_df[,1]
  
  
  #gene_counts_df[,1] <- NULL what is this doing
  # set what the factor of interest is
  if(idsBin == TRUE) {
    batch = metadata_df$idsBin
  }
  else {
    batch = metadata_df$scid_diagnosis
  }
  
  # correct for variance: if gene has no variance, remove it, is this correct? Should we be doing this????
  zeroVar = apply(gene_counts_df, 1, var)!=0
  for(gene in rownames(gene_counts_df)) {
    if(zeroVar[gene] == FALSE) {
      gene_counts_df[gene,] <- NA
    }
  }
  # remove all genes with 0 variance
  gene_counts_df <- gene_counts_df[complete.cases(gene_counts_df),]
  
  # run combat
  modcombat = model.matrix(~1, data = metadata_df)
  combat_edata = ComBat(dat = as.matrix(gene_counts_df), batch, modcombat)
  
  write.csv(combat_edata, file = writeName)

  combatResult = as.data.frame(combat_edata)
  
}