import pandas as pd
import subprocess
import os
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

# TODO: Run KBET should work inside python????
def kbet(gene_counts_path, metadata_path, output_path, output_name):

    print('Gene Counts Path')
    print(gene_counts_path)
    print('Metadata path')
    print(metadata_path)

    command = 'Rscript'
    dir_path = os.path.dirname(os.path.realpath(__file__))

    path_to_script = os.path.join(dir_path, "kbet.R")

    # print('kbet THIRD ARGUMENT')
    # print(test_data[2])

    args = [gene_counts_path, metadata_path, output_path, 'kbet_test']

    # build command
    cmd = [command, path_to_script] + args
    print('he really said')
    try:
        x = subprocess.check_output(cmd, universal_newlines=True, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    print('result of x', x)

    # now let's get the mean rejection rate
    #kBetResultsName = output_dir + '/' + test_data[2] + '_kBET_results.csv'
    # first let's get the csv
    #kBet_results = pd.read_csv(kBetResultsName)
    #rejection_rate = kBet_results.ix[0, 2]
    #acceptance_rate = 1 - rejection_rate

    return 1

    #return acceptance_rate


def getMetrics(test_data):

    print(test_data[2])

    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/' + test_data[2]
    try:
        os.mkdir(output_dir)
    except:
        print('it already exists okay')
        # except FileExistsError:
        # otherwise we're fine it already exists

    command = 'Rscript'
    path_to_script = "C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/kBET/kBET_script_pipeline.R"

    #print('kbet THIRD ARGUMENT')
    #print(test_data[2])

    args = [test_data[0], test_data[1], output_dir, test_data[2]]

    # build command
    cmd = [command, path_to_script] + args

    #NEED TO UNCOMMNE THIS

    #x = subprocess.check_output(cmd, universal_newlines=True, shell=True)

    #print(x)


    # now let's get the mean rejection rate
    kBetResultsName = output_dir + '/' + test_data[2] + '_kBET_results.csv'
    # first let's get the csv
    kBet_results = pd.read_csv(kBetResultsName)

    rejection_rate = kBet_results.ix[0, 2]
    acceptance_rate = 1 - rejection_rate

    print(test_data[2])
    print('kBET acceptance ' + str(acceptance_rate))
    print('Avedist ' + str(kBet_results.ix[0,4]))
    print('Pvcam ' + str(kBet_results.ix[0,5]))
    print('skewdiv ' + str(kBet_results.ix[0,6]))
    print('kldist ' + str(kBet_results.ix[0,7]))
    print()

    rejection_rate = kBet_results.ix[0, 2]
    acceptance_rate = 1 - rejection_rate

    return acceptance_rate


'''
 .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. |
| |  ____  ____  | || | ____   ____  | || |    ______    | |
| | |_   ||   _| | || ||_  _| |_  _| | || |  .' ___  |   | |
| |   | |__| |   | || |  \ \   / /   | || | / .'   \_|   | |
| |   |  __  |   | || |   \ \ / /    | || | | |    ____  | |
| |  _| |  | |_  | || |    \ ' /     | || | \ `.___]  _| | |
| | |____||____| | || |     \_/      | || |  `._____.'   | |
| |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------' 
'''
def makeTables(geneCountFile, annotFile):
    # read the annotation file
    annotTable = pd.read_csv(annotFile, sep=",", header=0, index_col=0)
    # read the gene counts file
    countTable = pd.read_csv(geneCountFile, sep=",", header=0, index_col=0).T
    countTable.index = countTable.index.map(int)  # index is messed up for some reason
    return (annotTable, countTable)


def sepBatches(countTable, annotTable, batchNum):
    # get meta_data
    # both_meta_data= pd.read_csv('C:/Users/Roman/Documents/Work/Depression_and_Immunology/Scripts/gender_analysis/meta_data/second_batch_mdd_both.csv').set_index('record_id')
    batchData = annotTable.loc[annotTable['batch'] == batchNum]
    #print(countTable.T.index.values)
    #print(countTable)
    return ((countTable).ix[batchData.index.values])

#    print(male_meta_data.index.values)

    # get both genders from second batch
    # both_gender_counts = gene_counts.ix[both_meta_data.index.values]
    # print(both_gender_counts.shape)
    # get only males

    #male_gender_counts = gene_counts.ix[male_meta_data.index.values]
'''
    #print('getting stuff')
    batchone = annotTable.loc[annotTable['batch'] == batchNum]
    #print('gotttemmmmm')
    ids = batchone.index.tolist()

    countTable.columns = countTable.columns.map(int)
    print(countTable)
    genebatch = countTable.iloc[:,countTable.columns.isin(ids)]
    print(countTable)
    return genebatch
'''

def topVarience(genebatch, percentage, name):
    #print(genebatch)
    output_dir = 'C:/Users/Roman/Documents/Work/Depression_and_Immunology/Spring_Research/Results/HVG_Vars/' + name
    genebatch = genebatch.T
    genebatch['varience'] = genebatch.var(axis=1).tolist() # check the AXIS

    variSort = genebatch.sort_values('varience', ascending=False)
    topGenes = len(variSort.index) * percentage
    variSortTop = variSort.head(math.floor(topGenes))
    if('combined_dataset_unfiltered' in name and '1' in name):
        print('PRINTING ' + str(name))
        genebatch.to_csv(output_dir + 'ORIGINAL_GENES.csv')
        #print(genebatch)
        print()

    #print(variSortTop['varience'])
    #variSortTop['varience'].to_csv(output_dir + 'ORIGINAL_GENES.csv')

    #print(variSortTop)
    return variSortTop.index


def getHvgs(countTableF, annotTableF, name):
    #print('GETTING HVGS FOR NAME ' + name)

    # does the file already exist?
    fileName = name + '_hvgs.txt'
    hvgs = []
    # IF THE FILE EXISTS
    if( False ): #os.path.isfile(fileName)): *** CHANGE THIS
        with open(fileName) as f:
            content = f.readlines()
        hvgs = [line.strip() for line in content]
        print('THE NUMBER OF HVGS FOR ' + str(name) + ' is ' + str(len(hvgs)))
        return hvgs
    # IF IT DOESN'T EXIST
    else:
        annotTable, countTable = makeTables(countTableF, annotTableF)
        numBatches = np.amax(annotTable['batch'])
        #print("made it this far")
        batches = [sepBatches(countTable, annotTable, batchNum) for batchNum in range(1, numBatches + 1)]
        #print('*** TOP VAR FOR ' + name)
        batchVariances = [topVarience(batch, .1, name + str(i)) for i, batch in enumerate(batches) ]
        hvgs = batchVariances[0]

        for i in range(1, len(batchVariances)):
                # something might be wrong here idk??????????
            #   print(str(0) + " combined with  " + str(i))
            hvgs = hvgs.intersection(batchVariances[i])
        allInterList = hvgs.tolist()
        with open(name + '_hvgs.txt', 'w') as f:
            for item in allInterList:
                f.write(item)
                f.write("\n")
        return allInterList

    #print("the number of hvgs is " + str(len(allInterList)))


def getIntersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def getRetained(quantativeData):
    #print(quantativeData)
    beforeCorrectionHvgs = []
    newQuantativeData = []
    correction = ['z_score', 'combat', 'limma', 'pca_loading']

    for data in quantativeData:
        originalData = True
        name = data[0]
        hvgs = data[1]

        # get the original hvgs
        for c in correction:
            if(c in name):
                originalData = False

        if(originalData):
            beforeCorrectionHvgs.append((name, hvgs))

    # go through again and get the percent of HVGs retained

    print('Before correction hvgs')
    for iterObject in beforeCorrectionHvgs:
        print(iterObject[0])

    for data in quantativeData:
        name = data[0]
        hvgs = data[1]
        acceptanceRate = data[2]
        # iterate through the before correction hvgs
        for originalData in beforeCorrectionHvgs:
            originalDataName = originalData[0]
            originalDataHvgs = originalData[1]

            if(originalDataName in name):
                print('Oirignal data hvgs' + str(originalDataName))
                #print(originalDataHvgs)
                print('this HVGS name ' + str(name))
                #print(hvgs)
                divisor = len(originalDataHvgs)
                if(len(originalDataHvgs) == 0):
                    divisor = 1
                # then this is the dataset it belongs to and we should calcualte the hvg retained

                # should we now do a rank analysis????
                percentRetained = (len(getIntersection(originalDataHvgs, hvgs)) / divisor) * 100.0

                print('Name ' + name)
                print('before correction hvgs')
                print(beforeCorrectionHvgs)
                print("orignal HVGS " + str(len(originalDataHvgs)))
                print('new hvgs ' + str(len(hvgs)))
                print('intersection is ' + str(len(getIntersection(originalDataHvgs, hvgs))))
                print('divisor ' + str(len(originalDataHvgs)))
                print('Percent retained ' + str(percentRetained))
                print()

                newQuantativeData.append( (name, acceptanceRate, percentRetained) )
                #break

    return newQuantativeData
