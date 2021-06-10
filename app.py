import os
from flask import Flask, render_template,request,redirect, url_for, flash,send_from_directory,session
from werkzeug.utils import secure_filename
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.secret_key = 'Sup3r_53cre7_k3y!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.static_folder = 'static'
#app.add_url_rule(
#    "/upload/<name>", endpoint="download_file", build_only=True
#)

#upload

#@app.route('/upload/<name>')
#def download_file(name):
#    return send_from_directory(app.config["UPLOAD_FOLDER"], name)




@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('download_file', name=filename))
            session['filename'] = filename  
            return redirect(url_for('newsample'))
    return render_template('upload.html').encode(encoding='UTF-8')
    

@app.route('/')
def index():
    return render_template('index.html').encode(encoding='UTF-8')

@app.route('/about')
def about():
    return render_template('about.html').encode(encoding='UTF-8')

@app.route('/example')
def example():
    
    #libraries
    from networkx.algorithms import community
    from bokeh.models import BoxSelectTool
    from bokeh.io import curdoc
    from bokeh.resources import CDN, INLINE
    from bokeh.themes import Theme
    

    import math
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import seaborn as sns
    import csv
    import numpy as np
    from bokeh.io import output_file, show
    from bokeh.layouts import gridplot, column,row
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, Slider, GraphRenderer, Ellipse, MultiLine, Range1d
    from bokeh.palettes import Spectral6, Spectral8, Viridis8
    from bokeh.transform import factor_cmap
    from scipy.interpolate import interp1d
    from scipy.ndimage.filters import gaussian_filter1d
    from scipy import stats
    from bokeh.io import curdoc
    from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                            MultiLine, Plot, Range1d, ResetTool,TapTool,NodesAndLinkedEdges, BoxSelectTool,EdgesAndLinkedNodes, WheelZoomTool, ResetTool)
    from bokeh.palettes import Spectral4, Paired8
    from bokeh.plotting import from_networkx
    from networkx.algorithms import community
    from bokeh.io import output_notebook, show, save
    from networkx.algorithms.community import greedy_modularity_communities
    output_notebook()

    #visualizations

    #reading
    df_enron = pd.read_csv("enron-v1.csv") 

    #1st vis

    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df_enron, source = 'fromId', target ='toId', create_using=Graphtype, edge_attr = 'sentiment', edge_key = 'sentiment')


    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name='degree', values=degrees)
    number_to_adjust_by = 5
    adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in nx.degree(G)])
    nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
    nx.set_node_attributes(G, name ='Sentiment', values = df_enron['sentiment'])
    nx.set_node_attributes(G, name = 'Email', values = df_enron['fromEmail'])
    nx.set_node_attributes(G, name = 'Title', values = df_enron['fromJobtitle'])
    communities = community.greedy_modularity_communities(G)
    GOOD_SENTIMENT, BAD_SENTIMENT , NEUTRAL= "cyan", "red","#C0C0C0"

    edge_attrs = {}

    for start_node, end_node , _  in G.edges(data=True):
        edge_color = GOOD_SENTIMENT if G.nodes[start_node]['Sentiment'] > 0 and G.nodes[end_node]['Sentiment'] > 0 else  BAD_SENTIMENT
        edge_attrs[(start_node, end_node)] = edge_color
        if G.nodes[start_node]['Sentiment'] == 0 and G.nodes[end_node]['Sentiment'] == 0 : edge_color = NEUTRAL
        edge_attrs[(start_node, end_node)] = edge_color


    nx.set_edge_attributes(G, edge_attrs, "edge_color")

    modularity_class = {}
    modularity_color = {}
    for community_number, community in enumerate(communities):
        for name in community: 
            modularity_class[name] = community_number
            modularity_color[name] = Paired8[community_number]

    nx.set_node_attributes(G, modularity_class, 'modularity_class')
    nx.set_node_attributes(G, modularity_color, 'modularity_color')


    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'modularity_color'
    color_palette = Viridis8

    node_hover_tool = HoverTool(tooltips=[("ID", "@index"), ("#MailsReciv", "@degree") , ("Email", "@Email"), ("Job Title", "@Title")])

    p = figure(tooltips = [("ID", "@index"), ("#MailsReciv", "@degree") , ("Email", "@Email"), ("Job Title", "@Title")],
                tools="hover,box_zoom,reset,save,box_select,pan,wheel_zoom, tap", active_scroll='wheel_zoom',
                x_range=Range1d(-1, 1), y_range=Range1d(-1, 1), plot_width = 700)


    p.title.text = "Graph for Sentiments between Reciever and Sender"
    p.toolbar.autohide = True



    graph_renderer = from_networkx(G, nx.kamada_kawai_layout, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.85, line_width=0.40 )
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color='yellow', line_width=2)

    graph_renderer.selection_policy = NodesAndLinkedEdges()

    p.renderers.append(graph_renderer)

    #2nd vis

    df_enron.sort_values(by=['date'], inplace = True)
    date = df_enron.groupby(df_enron['date']).count()
    sentiment = df_enron.groupby(df_enron['date']).mean()
    df_date = date.reset_index()
    df_sentiment = sentiment.reset_index()
    df_range= df_sentiment[['date']]
    df_range['count'] = df_date['fromId']
    df_range['senti'] = df_sentiment['sentiment']

    ranges=df_range['date']

    source = ColumnDataSource(df_range)
    tool_list = 'box_select', 'reset', 'wheel_zoom', 'pan', 'hover','tap'

    TOOLTIPS = [
        ("index", "$index"),
        ("Date", "@date"),
        ("Daily Count", "@count"),
    ]

    p1 = figure(x_range=ranges, plot_height=300, plot_width=750, toolbar_location='below', 
                title="Frequencey of messages by Date", tools=tool_list, 
                active_scroll='wheel_zoom', tooltips=TOOLTIPS)
    p1.vbar(x='date', top='count', width=0.9, source=source)
    p1.toolbar.autohide = True
    p1.xaxis.visible = False

    TOOLTIPS1 = [
        ("index", "$index"),
        ("Date", "@date"),
        ("Average Sentiment", "@senti"),
    ]

    p2 = figure(x_range=ranges, plot_height=300, plot_width=750, toolbar_location='below', 
                title=" Average Sentiment by Date", tools=tool_list, tooltips=TOOLTIPS1,
                active_scroll='wheel_zoom')
    p2.vbar(x='date', top='senti', width=0.9, source=source, color="orange")
    p2.toolbar.autohide = True
    p2.xaxis.visible = False

    p1.xgrid.grid_line_color = None
    p1.y_range.start = 0
    p1.y_range.end = 200
    p2.xgrid.grid_line_color = None

    plot_f = column(p1, p2)

    plot = row(p, plot_f)


    #curdoc().theme = 'dark_minimal'
    curdoc().add_root(plot)
    curdoc().theme = Theme(filename='theme.yaml')

    #site embedding

    script1, div1,=components(plot)
    cdn_js = INLINE.render_js()
    cdn_css = INLINE.render_css() 
    return render_template('example.html',
    script1=script1,
    div1=div1, 
    cdn_js=cdn_js,
    cdn_css=cdn_css).encode(encoding='UTF-8')#render_template("example.html",script1=script1,div1=div1,cdn_js=cdn_js,cdn_css=cdn_css)




@app.route('/sources')
def sources():
    return render_template('sources.html').encode(encoding='UTF-8')

@app.route('/first_visualisation')
def first_visualization():
    return 'First visualisation  &nbsp; <a href="/">Home</a>'

@app.route('/second_visualisation')
def second_visualization():
    return 'Second visualisation &nbsp; <a href="/">Home</a>'

@app.route('/newsample')
def newsample():
    #callback last session for dataset
    if 'filename' in session : 
        filename = session['filename']
    else:
        filename = 'sample_b.csv'

    
    #libraries
    from networkx.algorithms import community
    from bokeh.models import BoxSelectTool
    from bokeh.io import curdoc
    from bokeh.resources import CDN, INLINE
    from bokeh.themes import Theme
    

    import math
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import seaborn as sns
    import csv
    import numpy as np
    from bokeh.io import output_file, show
    from bokeh.layouts import gridplot, column,row
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, Slider, GraphRenderer, Ellipse, MultiLine, Range1d
    from bokeh.palettes import Spectral6, Spectral8, Viridis8
    from bokeh.transform import factor_cmap
    from scipy.interpolate import interp1d
    from scipy.ndimage.filters import gaussian_filter1d
    from scipy import stats
    from bokeh.io import curdoc
    from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                            MultiLine, Plot, Range1d, ResetTool,TapTool,NodesAndLinkedEdges, BoxSelectTool,EdgesAndLinkedNodes, WheelZoomTool, ResetTool)
    from bokeh.palettes import Spectral4, Paired8
    from bokeh.plotting import from_networkx
    from networkx.algorithms import community
    from bokeh.io import output_notebook, show, save
    from networkx.algorithms.community import greedy_modularity_communities
    output_notebook()

    #visualizations



    #reading
    df_enron = pd.read_csv("upload/"+filename) 

    #1st vis

    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df_enron, source = 'fromId', target ='toId', create_using=Graphtype, edge_attr = 'sentiment', edge_key = 'sentiment')


    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name='degree', values=degrees)
    number_to_adjust_by = 5
    adjusted_node_size = dict([(node, degree+number_to_adjust_by) for node, degree in nx.degree(G)])
    nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
    nx.set_node_attributes(G, name ='Sentiment', values = df_enron['sentiment'])
    nx.set_node_attributes(G, name = 'Email', values = df_enron['fromEmail'])
    nx.set_node_attributes(G, name = 'Title', values = df_enron['fromJobtitle'])
    communities = community.greedy_modularity_communities(G)
    GOOD_SENTIMENT, BAD_SENTIMENT , NEUTRAL= "cyan", "red","#C0C0C0"

    edge_attrs = {}

    for start_node, end_node , _  in G.edges(data=True):
        edge_color = GOOD_SENTIMENT if G.nodes[start_node]['Sentiment'] > 0 and G.nodes[end_node]['Sentiment'] > 0 else  BAD_SENTIMENT
        edge_attrs[(start_node, end_node)] = edge_color
        if G.nodes[start_node]['Sentiment'] == 0 and G.nodes[end_node]['Sentiment'] == 0 : edge_color = NEUTRAL
        edge_attrs[(start_node, end_node)] = edge_color


    nx.set_edge_attributes(G, edge_attrs, "edge_color")

    modularity_class = {}
    modularity_color = {}
    for community_number, community in enumerate(communities):
        for name in community: 
            modularity_class[name] = community_number
            modularity_color[name] = Paired8[community_number]

    nx.set_node_attributes(G, modularity_class, 'modularity_class')
    nx.set_node_attributes(G, modularity_color, 'modularity_color')


    size_by_this_attribute = 'adjusted_node_size'
    color_by_this_attribute = 'modularity_color'
    color_palette = Viridis8

    node_hover_tool = HoverTool(tooltips=[("ID", "@index"), ("#MailsReciv", "@degree") , ("Email", "@Email"), ("Job Title", "@Title")])

    p = figure(tooltips = [("ID", "@index"), ("#MailsReciv", "@degree") , ("Email", "@Email"), ("Job Title", "@Title")],
                tools="hover,box_zoom,reset,save,box_select,pan,wheel_zoom, tap", active_scroll='wheel_zoom',
                x_range=Range1d(-1, 1), y_range=Range1d(-1, 1),plot_width=700)


    p.title.text = "Graph for Sentiments between Reciever and Sender"
    p.toolbar.autohide = True



    graph_renderer = from_networkx(G, nx.kamada_kawai_layout, scale=1, center=(0, 0))

    graph_renderer.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.85, line_width=0.40 )
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color='yellow', line_width=2)

    graph_renderer.selection_policy = NodesAndLinkedEdges()

    p.renderers.append(graph_renderer)

    #2nd vis

    df_enron.sort_values(by=['date'], inplace = True)
    date = df_enron.groupby(df_enron['date']).count()
    sentiment = df_enron.groupby(df_enron['date']).mean()
    df_date = date.reset_index()
    df_sentiment = sentiment.reset_index()
    df_range= df_sentiment[['date']]
    df_range['count'] = df_date['fromId']
    df_range['senti'] = df_sentiment['sentiment']

    ranges=df_range['date']

    source = ColumnDataSource(df_range)
    tool_list = 'box_select', 'reset', 'wheel_zoom', 'pan', 'hover','tap'

    TOOLTIPS = [
        ("index", "$index"),
        ("Date", "@date"),
        ("Daily Count", "@count"),
    ]

    p1 = figure(x_range=ranges, plot_height=300, plot_width=750, toolbar_location='below', 
                title="Frequencey of messages by Date", tools=tool_list, 
                active_scroll='wheel_zoom', tooltips=TOOLTIPS)
    p1.vbar(x='date', top='count', width=0.9, source=source)
    p1.toolbar.autohide = True
    p1.xaxis.visible = False

    TOOLTIPS1 = [
        ("index", "$index"),
        ("Date", "@date"),
        ("Average Sentiment", "@senti"),
    ]

    p2 = figure(x_range=ranges, plot_height=300, plot_width=750, toolbar_location='below', 
                title=" Average Sentiment by Date", tools=tool_list, tooltips=TOOLTIPS1,
                active_scroll='wheel_zoom')
    p2.vbar(x='date', top='senti', width=0.9, source=source, color="orange")
    p2.toolbar.autohide = True
    p2.xaxis.visible = False

    p1.xgrid.grid_line_color = None
    p1.y_range.start = 0
    p1.y_range.end = 200
    p2.xgrid.grid_line_color = None

    plot_f = column(p1, p2)

    plot = row(p, plot_f)


    #curdoc().theme = 'dark_minimal'
    curdoc().add_root(plot)
    curdoc().theme = Theme(filename='theme.yaml')

    #site embedding

    script1, div1,=components(plot)
    cdn_js = INLINE.render_js()
    cdn_css = INLINE.render_css() 
    return render_template('newsample.html',
    script1=script1,
    div1=div1, 
    cdn_js=cdn_js,
    cdn_css=cdn_css).encode(encoding='UTF-8')

if __name__ == '__main__':
    app.run(debug=True)