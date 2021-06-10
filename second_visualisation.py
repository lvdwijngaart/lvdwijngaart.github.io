def second_visualisation():
    import math
    import pandas as pd
    import networkx as nx
    ot as plt
    import seaborn as sns
    utput_file, show
    ort gridplot, column,row
    port figure, show
    rt ColumnDataSource, Slider, GraphRenderer, Ellipse
    port Spectral6, Spectral8
    mport factor_cmap
     import interp1d
    ters import gaussian_filter1d
    s
    urdoc

    _minimal'

    ('enron-v1.csv')
    y=['date'], inplace = True)

    nron.groupby(df_enron['date']).count()
     df_enron.groupby(df_enron['date']).max()

    rouped_date.reset_index()
     df_grouped_sentiment.reset_index() 

    entiment[['date']]
    grouped_date['fromId']
    grouped_sentiment['sentiment']
    o_datetime(df_copy["date"]).dt.strftime("%Y%m%d").astype(int)

    sentiment[['date']]
    f_grouped_date['fromId']
    f_grouped_sentiment['sentiment']

    ve graphs.html")

    e']

    rce(df_column)
    t', 'reset', 'help', 'wheel_zoom', 'pan', 'hover'

    e=x_range, plot_height=250, toolbar_location='below',     essages by Date",tools="hover,box_zoom,reset,save,box_select, active_scroll = 'wheel_zoom', tooltips = [("Day", "@index")])
    top='count', width=0.9, source=source)

    e=x_range, plot_height=250, toolbar_location='below',     nt by Date",tools="hover,box_zoom,reset,save,box_select,  ctive_scroll = 'wheel_zoom',tooltips = [("Day", "@index")])
    top='senti', width=0.9, source=source, color="orange")
    l = 'Day in the whole time frame'
    l = 'Day in the whole time frame'
     False
     False

    _color = None
    = 0
    200
    _color = None

    lot_2)


    show(plot_f)

    return 0