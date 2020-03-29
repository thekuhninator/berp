from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.legend_handler import HandlerBase
from matplotlib.markers import MarkerStyle as Markers
import pandas as pd
import os

meta_colors = [
"#FF0000","#00FF00","#0000FF","#FFFF00","#00FFFF","#FF00FF",
"#808080","#FF8080","#80FF80","#8080FF","#008080","#800080","#808000","#FFFF80",
"#80FFFF","#FF80FF","#FF0080","#80FF00","#0080FF","#00FF80","#8000FF","#FF8000",
"#000080","#800000","#008000","#404040","#FF4040","#40FF40","#4040FF","#004040",
"#400040","#404000","#804040","#408040","#404080","#FFFF40","#40FFFF","#FF40FF",
"#FF0040","#40FF00","#0040FF","#FF8040","#40FF80","#8040FF","#00FF40","#4000FF",
"#FF4000","#000040","#400000","#004000","#008040","#400080","#804000","#80FF40",
"#4080FF","#FF4080","#800040","#408000","#004080","#808040","#408080","#804080",
"#C0C0C0","#FFC0C0","#C0FFC0","#C0C0FF","#00C0C0","#C000C0","#C0C000","#80C0C0",
"#C080C0","#C0C080","#40C0C0","#C040C0","#C0C040","#FFFFC0","#C0FFFF","#FFC0FF",
"#FF00C0","#C0FF00","#00C0FF","#FF80C0","#C0FF80","#80C0FF","#FF40C0","#C0FF40",
"#40C0FF","#00FFC0","#C000FF","#FFC000","#0000C0","#C00000","#00C000","#0080C0",
"#C00080","#80C000","#0040C0","#C00040","#40C000","#80FFC0","#C080FF","#FFC080",
"#8000C0","#C08000","#00C080","#8080C0","#C08080","#80C080","#8040C0","#C08040",
"#40C080","#40FFC0","#C040FF","#FFC040","#4000C0","#C04000","#00C040","#4080C0",
"#C04080","#80C040","#4040C0","#C04040","#40C040","#202020","#FF2020","#20FF20"]

'''
 .----------------.  .----------------.  .----------------.  .-----------------. .----------------.
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |  _________   | || |              | || |    _______   | || | ____  _____  | || |  _________   | |
| | |  _   _  |  | || |              | || |   /  ___  |  | || ||_   \|_   _| | || | |_   ___  |  | |
| | |_/ | | \_|  | || |    ______    | || |  |  (__ \_|  | || |  |   \ | |   | || |   | |_  \_|  | |
| |     | |      | || |   |______|   | || |   '.___`-.   | || |  | |\ \| |   | || |   |  _|  _   | |
| |    _| |_     | || |              | || |  |`\____) |  | || | _| |_\   |_  | || |  _| |___/ |  | |
| |   |_____|    | || |              | || |  |_______.'  | || ||_____|\____| | || | |_________|  | |
| |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------'  '----------------'
'''

# TODO: we need to make this to work for whatever variables they pass in...
def plot_tsne(gene_counts ,metadata, perplexity, factor, output_dir, plot_name = None):

    print('calling new tsne')
    # create the name of the save file
    save_file = os.path.join(output_dir, plot_name)

    # create the TSNE model
    model = TSNE(n_components=2, perplexity=perplexity, n_iter = 5000)
    model = model.fit_transform(gene_counts)

    # tsne stuff stuff

    tsne_df = pd.DataFrame()
    tsne_df['x'] = model[:, 0]
    tsne_df['y'] = model[:, 1]

    # TODO: this needs to take each as factors and then map them to colors
    # get all the different factors for the batch column
    batch_factors = list(metadata['batch'].unique())
    print('batch factors')
    print(batch_factors)
    # get all the different factors for the factor of interest column
    foi_factors = metadata[factor].unique()
    foi_factors = list(metadata[factor].value_counts(sort=True).index)
    print('factor of interets factors')
    print(foi_factors)


    # set everything past a certain value of foi_factors to other...
    if(len(foi_factors) > len(Markers.filled_markers)):
        # then let's set all to other past the length of filled markers
        foi_factors[len(Markers.filled_markers) - 1] = 'Other'
        foi_factors = foi_factors[0: len(Markers.filled_markers)]

    # map the foi factors to shapes
    marker_map = {foi_factors[i]: Markers.filled_markers[i] for i in range(len(foi_factors))}
    # now we have to map batch values to colors
    # now for colormap what we have to do is take a look at the actual column for batch, and then map each using color map
    # get the value back as a list and give it to the people i guess.
    color_map = {batch_factors[i]: meta_colors[i % len(meta_colors)] for i in range(len(batch_factors))}
    new_color_values = metadata['batch'].map(color_map)
    print(new_color_values)

    marker_labels = foi_factors
    color_labels = ['Batch ' + str(batch_factors[i]) for i in range(len(batch_factors))]

    list_markers = [marker for (factor, marker) in marker_map.items()]
    # we need to append circles for each batch
    list_markers = list_markers + ['o' for _ in range(len(color_labels))]

    list_colors = ['k' for _ in range(len(marker_labels))]
    list_colors = list_colors + [color for (batch, color) in color_map.items()]
    list_labels = marker_labels + color_labels

    # print variables
    print('Marker Map')
    print(marker_map)
    print('Color_Map')
    print(color_map)
    print('Marker lables')
    print(marker_labels)
    print('color lables')
    print(color_labels)

    print()
    print('list markers')
    print(list_markers)
    print('list colors')
    print(list_colors)
    print('list labels')
    print(list_labels)

    batch_values = metadata['batch'].map(color_map)
    foi_values   = metadata[factor].map(marker_map)

    # list labels... first should be factors of interests, then batch
    #list_lab = ['Healthy', 'Depression', 'Bipolar', 'Batch 1 (Toups)', 'Batch 2 (Toups)', 'Batch 3 (Toups)',
    #            'Batch 4 (DGN Dataset)']


    #batchValues = metadata['batch'].map({0.0: 'w', 1.0:'r', 2.0:'g', 3.0:'b', 4.0:'y'})
    # TODO: make it grab the factor of interest by itself
    # diagnosisValues = metadata[factor].map({0.0: 'P', 1.0:'d', 2.0:'^', 3.0:'^', 4.0:'P'})

    # create a list of colors and markers for however many levels exist
    # if there are too many then display an error
    # TODO: we need to figure out how we want to do this with t-sne...

    #list_color = ['k', 'k', 'k', 'r', 'g', 'b', 'y']
    #list_marker = ["P", "d", "^", "o", "o", "o", "o"]

    # for list lab we will have to grab the variables for
    # for list lab: first few are black, which represent the varaibles assigned to factor of interest
    # for list lab: next correspond to batch
    # list_lab = ['Healthy', 'Depression', 'Bipolar', 'Batch 1 (Toups)', 'Batch 2 (Toups)', 'Batch 3 (Toups)', 'Batch 4 (DGN Dataset)']



    # idk what this does
    ax = plt.gca()

    # set the background to white

    #ax.set_facecolor('#ffffff')
    #for spine in ax.spines.values():
    #    spine.set_visible(True)
    #    spine.set_color('red')

    # put this class outside or some shit

    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                           marker=tup[1],color=tup[0], transform=trans)]

    for x,y,b,d in zip(tsne_df['x'], tsne_df['y'], new_color_values, foi_values):
        ax.scatter(x, y, c=b, marker=d)

    ax.legend(list(zip(list_colors, list_markers)), list_labels, handler_map={tuple:MarkerHandler()})

    # save the t-sne file

    print('WEREA BOUT TO SAVE THE T-SNE FIGURE')
    print(output_dir + 't-sne_perplex_' + str(perplexity) + '_' + plot_name + '.png')

    plot_title = ' '.join([x.capitalize() for x in plot_name.split('_')])

    plt.title(plot_title + ' - T-SNE Perplexity: ' + str(perplexity))
    #plt.show()

    # create output path and save the figure
    output_dir = './'
    output_path = output_dir + 'tsne-figure.png'
    plt.savefig(output_path)
    #plt.savefig(output_dir + 't-sne_perplex_' + str(perplexity) + '_' + plot_name + '.png')
    #plt.clf()
    #plt.cla()
    #plt.close()

    return output_path