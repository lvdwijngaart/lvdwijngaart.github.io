def do_first_visualisation():
  import pandas as pd
  import networkx
  t as plt
  import numpy as np
  import math
  import pandas as pd
  import networkx as nx
  t as plt
  import seaborn as sns
  tput_file, show
  rt gridplot, column
  ort figure, show
  t ColumnDataSource, Slider, GraphRenderer, Ellipse
  ort Spectral6, Spectral8
  port factor_cmap
  import interp1d
  ers import gaussian_filter1d
  from scipy import stats
  tput_notebook, show, save
  t Range1d, Circle, ColumnDataSource, MultiLine
  ort figure
  ort from_networkx
  ort Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
  port linear_cmap
  s import community
  tput_file, show
  rt gridplot
  t ColumnDataSource
  ort figure
  import pandas as pd
   components
  ort d3
  ort figure
  import seaborn as sns
  t ColumnDataSource, Slider
  import pandas as pd
  import networkx as nx
  tput_file, show
  rt gridplot, column
  ort figure, show
  t ColumnDataSource, Slider
  tput_notebook, show, save
  t Range1d, Circle, ColumnDataSource, MultiLine
  ort figure
  ort from_networkx
  ort Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
  port linear_cmap
  s import community
  tput_file, show
  rt gridplot
  t ColumnDataSource
  ort figure
  import networkx as nx
  import pandas as pd
  import seaborn as sns
  import pandas as pd
  t as plt
  import networkx as nx
  import csv
  import math
  import pandas as pd
  import networkx
  t as plt
  import numpy as np
  tput_notebook, show, save
  output_notebook()

  "enron-v1.csv") 
  df_enron

  tput_file, show
  t (BoxZoomTool, Circle, HoverTool,
     MultiLine, Plot, Range1d, ResetTool,TapTool,NodesAndLinkedEdges, BoxSelectTool,EdgesAndLinkedNodes, WheelZoomTool, ResetTool)
  ort Spectral4, Paired8
  ort from_networkx
  s import community
  t BoxSelectTool

  rdoc

  minimal'

  "enron-v1.csv") 


  Graphtype = nx.Graph()
  list(df_enron, source = 'fromId', target ='toId',   edge_attr = 'sentiment', edge_key = 'sentiment')




  e(G))
  G, name='degree', values=degrees)
  number_to_adjust_by = 5
  ct([(node, degree+number_to_adjust_by) for node, degree in   
  G, name='adjusted_node_size', values=adjusted_node_size)
  G, name ='Sentiment', values = df_enron['sentiment'])
  G, name = 'Email', values = df_enron['fromEmail'])
  G, name = 'Title', values = df_enron['fromJobtitle'])
  .greedy_modularity_communities(G)
  TIMENT , NEUTRAL= "cyan", "red","#C0C0C0"

  edge_attrs = {}

  de , _  in G.edges(data=True):
  ENTIMENT if G.nodes[start_node]['Sentiment'] > 0 and G.nodes  > 0 else BAD_SENTIMENT
  ode, end_node)] = edge_color
  e , _  in G.edges(data=True):
  NTIMENT if G.nodes[start_node]['Sentiment'] > 0 and G.nodes  t'] > 0 else  BAD_SENTIMENT
  de, end_node)] = edge_color
  e]['Sentiment'] == 0 and G.nodes[end_node]['Sentiment'] == 0   AL
  de, end_node)] = edge_color


  G, edge_attrs, "edge_color")

  modularity_class = {}
  modularity_color = {}
  ommunity in enumerate(communities):
  ty: 
  s[name] = community_number
  r[name] = Paired8[community_number]

  G, modularity_class, 'modularity_class')
  G, modularity_color, 'modularity_color')


  = 'adjusted_node_size'
   = 'modularity_color'
  8

  Tool(tooltips=[("ID", "@index"), ("#MailsReciv", "@degree")  ("Job Title", "@Title")])

  = [("ID", "@index"), ("#MailsReciv", "@degree") , ("Email",  , "@Title")],
  ver,box_zoom,reset,save,box_select,pan,wheel_zoom, tap",   roll='wheel_zoom',
  ge1d(-1, 1), y_range=Range1d(-1, 1))


  h for Sentiments between Reciever and Sender"



  etworkx(G, nx.kamada_kawai_layout, scale=1, center=(0, 0))

  derer.glyph = Circle(size=size_by_this_attribute,  s_attribute)

  derer.glyph = MultiLine(line_color="edge_color",   idth=0.40 )
  derer.selection_glyph = MultiLine(line_color='yellow',   

  n_policy = NodesAndLinkedEdges()

  raph_renderer)

  e Nodes.html")

  show(plot)



  #OVER_TOOLTIPS1 = [
  @index"),
  #]

  ver,box_zoom,reset,save,box_select,pan,wheel_zoom, tap",   S1 )
  'fromId'] , df_enron['toId'], radius=0.2, alpha=0.5)

  .1, end=2, step=0.01, value=0.2)
   r.glyph, 'radius')

  rce(data = df_enron)

  import numpy as np

  ort figure, output_file, show
  rt gridplot, column, row

  N = 4000
  ze=N) * 100
  ze=N) * 100
  m(size=N) * 1.5
  ,g, 150] for r,g in zip(50 + 2*x, 30 + 2*y) ], dtype="uint8")

  list(df_enron, source = 'fromId',target = 'toId',   edge_attr = 'sentiment', edge_key = 'sentiment')


  ("ID", "@index"), ("#MailsReciv", "@degree") , ("Email",   , "@Title")],
  ver,box_zoom,reset,save,box_select,pan,wheel_zoom, tap",   roll='wheel_zoom',
  ge1d(-10.1, 10.1),y_range = Range1d(-10.1, 10.1))

  networkx(S, nx.spring_layout, scale=1, center=(0, 0))
  nderer.glyph = Circle(fill_alpha=0.5, fill_color = 'red')
  nderer.glyph = MultiLine(line_color="edge_color",  dth=0 )
  h_renderer1)

  r,box_zoom,reset,save,box_select,pan,wheel_zoom, tap",   om'),
  nge1d(-10.1, 10.1),y_range = Range1d(-10.1, 10.1))
  1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
  _tool, BoxZoomTool(), HoverTool(tooltips=None), TapTool(),   oomTool(), ResetTool())
  'fromId'], df_enron['toId'],
  6,
  e)
  _networkx(S, nx.spring_layout, scale=1, center=(0, 0))
  enderer.glyph = Circle(fill_alpha=0.5, fill_color = 'red')
  enderer.glyph = MultiLine(line_color="edge_color",   dth=0 )
  ph_renderer1)

            
  olor = None
  #plot.y_range.start = 0
  #plot.y_range.end = 200
  r = None

  400, plot_height=400, tools="hover,box_zoom,reset,save,  om, tap", active_scroll='wheel_zoom')#, x_range=Range1d(-10. 1d(-10.1, 10.1))
  mId'], df_enron['toId'], radius=radii, alpha=0.5)

  e_graphs.html")
  x = row(plot, p)
  show(x)

  return 0
