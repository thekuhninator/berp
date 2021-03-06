#! /usr/bin/Rscript

# install k-bet

#library(devtools)
options(repos = getOption("repos")["CRAN"])
list.of.packages <- c("devtools", "remotes", "BiocManager", "bapred", "ggplot2")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) {
  print('installing')
  install.packages(new.packages, repos = "http://cran.us.r-project.org", dependencies= TRUE)
}

library(devtools)

if (!requireNamespace("kBET", quietly = TRUE))
{
  BiocManager::install('theislab/kBET', dependencies= TRUE)
  
}


BiocManager::install('bapred') 

if(!require('sva'))
{
  BiocManager::install('sva')
}
if(!require('BiocParallel'))
{
  BiocManager::install('BiocParallel')
}
if(!require('affyPLM'))
{
  BiocManager::install('affyPLM')
}



library(BiocManager)
#if(!require(kBET)) {
if (!requireNamespace("kBET", quietly = TRUE))
{
  BiocManager::install('theislab/kBET', dependencies= TRUE)

}

  
#}
library(kBET)
library(bapred)
library(ggplot2)

args = commandArgs(trailingOnly=TRUE)

# Collect file name from user
#input_counts <- readline(prompt = "What is the name of your gene counts file?: ")
input_counts = args[1]
input_annot  = args[2]
output_dir = args[3]
output_name = args[4]


# what is this code doing...
temp = unlist(strsplit(input_annot, "metadata"))[1]
name = unlist(strsplit( temp ,  "[\\\\]|[^[:print:]]") )
name = name[length(name)]


# Read in data file
my_data <- as.data.frame(t(read.csv(input_counts, header = TRUE, row.names = 1, check.names=FALSE)))

#Read in annotation file
annot <- read.csv(input_annot, header = TRUE, row.names = 1, check.names = FALSE)
batch = annot$batch
# change this to be factor of interest...
diagnosis = annot$scid_diagnosis

# not quite sure what this is doing... get rid of zero variance genes
new_data = my_data[ ,which(apply(t(my_data), 1, var) != 0)]

print('ouput name right now')
print(name)

print('output dir')
print(output_dir)

print('Batch')
print(dim(batch))

print('Data')
print(dim(my_data))
print(dim(new_data))



#my_data

#data: a matrix (rows: samples, columns: features (genes))
#batch: vector or factor with batch label of each cell 

#rownames(annot[annot['batch'] == 2,])
#rownames(data.frame(gene_df))
#gene_df[]
#new_gene_df <- my_data[rownames(annot[annot['batch'] == 1 | annot['batch'] == 1,]),]

#new_gene_df

#batch2 = annot[annot['batch'] == 1 | annot['batch'] == 1,]$batch


avedistVal = avedist(new_data, as.factor(batch) )#, as.factor(diagnosis))
pvcamVal = pvcam(new_data, as.factor(batch), as.factor(diagnosis))
skewdivVal = skewdiv(new_data, as.factor(batch))
#sepscoreVal = sepscore(new_data, as.factor(batch) )
kldistVal = kldist(new_data, as.factor(batch))
# diffexprmVal = 
# cobraVal = 
batch.estimate <- kBET(my_data, batch, plot = FALSE)
#batch.estimate$results
#batch.estimate


#batch.estimate$summary

# Capitalize Function
capitalize <- function(x) {
  substr(x, 1, 1) <- toupper(substr(x, 1, 1))
  x
}

# Generate the Plot title
plotTitle = paste(capitalize(unlist(strsplit(name, "_"))), collapse= " ")

png(file=paste( output_dir,'/', output_name, ".png", sep=""))
#batch.estimate <- kBET(data, batch, plot=FALSE)
plot.data <- data.frame(class=rep(c('observed', 'expected'), 
                                  each=length(batch.estimate$stats$kBET.observed)), 
                        data =  c(batch.estimate$stats$kBET.observed,
                                  batch.estimate$stats$kBET.expected))
g <- ggplot(plot.data, aes(class, data)) + geom_boxplot() + 
     labs(x='Test', y='Rejection rate',title=paste(plotTitle,'kBET test results',sep=' ')) +
     theme_bw() +  
     scale_y_continuous(limits=c(0,1))

g

dev.off()
name

results <- batch.estimate$summary
results['avedist'] = avedistVal
results['pvcam'] = pvcamVal
results['skewdiv'] = skewdivVal
results['kldist'] = kldistVal

results

print('*** DONE WRITING NOW')

write.csv(results, file=paste(output_dir,'/', output_name, "_kBET_results.csv", sep=''))

#dev.cur()