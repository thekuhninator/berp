import base64

'''
with open("my_image.jpg", "rb") as img_file:
    my_string = base64.b64encode(img_file.read())
print(my_string)
'''
def generate_report(output_name, dataset_name, boxplot_path, tsne_path):
    print('Paths for report')
    print(boxplot_path)
    print(tsne_path)
    html_string = '''
        <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <style>body{ margin:0 100; background:whitesmoke; }</style>
        </head>
        <body>
            <h1>Batch Correction Report for ''' + dataset_name + '''</h1>
    
            <!-- *** Section 1 *** --->
            <h2>T-SNE Report</h2>
             <iframe width="1000" height="550" frameborder="0" seamless="seamless" scrolling="no" \
src="''' + tsne_path + '''"></iframe>
            <p>T-SNE is a t-distributed stochastic neighbor estimated.</p>
            
            <!-- *** Section 2 *** --->
            <h2>Comparative BoxPlot</h2>
             <iframe width="1000" height="550" frameborder="0" seamless="seamless" scrolling="no" \
src="''' + boxplot_path + '''"></iframe>

            <p>The comparative boxplot shows the difference in the distributions between the different batches. If any of the boxes differ significanlty then there is a difference.</p>
        </body>
    </html>
    '''
    f = open(output_name, 'w')
    f.write(html_string)
    f.close()

    # should we automatically add .html to the end if they don't specify
    # yeah probably

    # get the image files to generate and stuff then read them in in this function and then use them to save the data
    # internally within the package.

