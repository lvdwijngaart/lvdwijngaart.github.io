#libraries

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
            x_range=Range1d(-1, 1), y_range=Range1d(-1, 1))


p.title.text = "Graph for Sentiments between Reciever and Sender"



graph_renderer = from_networkx(G, nx.kamada_kawai_layout, scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)

graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.85, line_width=0.40 )
graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color='yellow', line_width=2)

graph_renderer.selection_policy = NodesAndLinkedEdges()

p.renderers.append(graph_renderer)

#2nd vis

df_enron.sort_values(by=['date'], inplace = True)

df_grouped_date = df_enron.groupby(df_enron['date']).count()
df_grouped_sentiment = df_enron.groupby(df_enron['date']).mean()

df_grouped_date = df_grouped_date.reset_index()
df_grouped_sentiment = df_grouped_sentiment.reset_index() 

df_copy = df_grouped_sentiment[['date']]
df_copy['count'] = df_grouped_date['fromId']
df_copy['senti'] = df_grouped_sentiment['sentiment']
df_copy["date"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y%m%d").astype(int)

df_column= df_grouped_sentiment[['date']]
df_column['count'] = df_grouped_date['fromId']
df_column['senti'] = df_grouped_sentiment['sentiment']

output_file("Interactive graphs.html")

x_range=df_column['date']

source = ColumnDataSource(df_column)
tool_list = 'box_select', 'reset', 'help', 'wheel_zoom', 'pan', 'hover'

p1 = figure(x_range=x_range, plot_height=250, toolbar_location='below', title="Frequencey of messages by Date",tools="hover,box_zoom,reset,save,box_select,wheel_zoom, tap,pan", active_scroll = 'wheel_zoom', tooltips = [("Day", "@index")])
p1.vbar(x='date', top='count', width=0.9, source=source)

p2 = figure(x_range=x_range, plot_height=250, toolbar_location='below', title="Maximum Sentiment by Date",tools="hover,box_zoom,reset,save,box_select,wheel_zoom, pan,tap",active_scroll = 'wheel_zoom',tooltips = [("Day", "@index")])
p2.vbar(x='date', top='senti', width=0.9, source=source, color="orange")
p1.xaxis.axis_label = 'Day in the whole time frame'
p2.xaxis.axis_label = 'Day in the whole time frame'
p1.xaxis.visible = False
p2.xaxis.visible = False

p1.xgrid.grid_line_color = None
p2.xgrid.grid_line_color = None

plot_f = row(p1, p2)

plot = column(p, plot_f)
show(plot)