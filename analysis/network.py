from graph_tool.all import arf_layout, Graph, graph_draw, get_hierarchy_tree, radial_tree_layout, \
    get_hierarchy_control_points, minimize_nested_blockmodel_dl

import numpy as np
import matplotlib.pyplot as plt

import imageio
from tqdm.notebook import tqdm

def state_to_color(state, max_value):
    # Normalize the state value to between 0 and 1
    normalized_state = state / max_value

    # Get the "cividis" colormap
    cmap = plt.get_cmap("viridis")

    # Convert the normalized state value to a color
    color = cmap(normalized_state)

    return list(color)

def create_network(W, Win=None, Wout=None):
    # Size of the number of connection in the connection matrice
    row, col = W.row, W.col
    W = W.toarray() / np.amax(W)
    n = len(W)
    
    # Initialize the graph
    g = Graph(directed=True)
    
    # We add the EDGES properties
    # For other properties : https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.draw.graph_draw.html#graph_tool.draw.graph_draw    
    # Other types : https://graph-tool.skewed.de/static/doc/quickstart.html#property-maps
    # Graph object doc : https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.Graph.html#graph_tool.Graph.vp
    
    # edges index and width
    edge_index = []
    pen_width = []
    
    for i in range(len(col)):
        edge_index.append((row[i], col[i]))
        pen_width.append(W[row[i]][col[i]])

    if Win is not None:
        Win = Win.toarray() / np.amax(Win)
        for i in range(n):
            edge_index.append((n, i))
            pen_width.append(Win[i][0])

    if Wout is not None:
        Wout = Wout.toarray() / np.amax(Wout)
        for i in range(n):
            edge_index.append((i, n + 1))
            pen_width.append(Wout[i])
            
    g.add_edge_list(np.array(edge_index))
    edge_pen_width = g.new_edge_property("double")
    edge_pen_width.a = np.abs(pen_width) * 3

    # curvatures of the edges
    state = minimize_nested_blockmodel_dl(g)
    t = get_hierarchy_tree(state)[0]
    tpos = radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
    cts = get_hierarchy_control_points(g, t, tpos)
    
    # add those properties
    g.properties[("e", "pen_width")] = edge_pen_width
    g.properties[("e", "cts")] = cts

    #We add the VERTEX properties
    # For other properties : https://graph-tool.skewed.de/static/doc/autosummary/graph_tool.draw.graph_draw.html#graph_tool.draw.graph_draw
        
    #vertex text
    text = g.new_vp("string")
    for v in g.vertex_index:
        text[v] = str(v)

    #vertex position
    pos= g.new_gp("vector<float>")
    pos = arf_layout(g)
    # Position of the output node
    pos[n + 1] = [7, 4]
    # Position of the input node
    pos[n] = [1, 4]
    
    # add the properties
    g.properties[("v", "text")] = text
    g.properties[("v", "pos")] = pos
    
    return g

    
def draw_network(g, filename=None):
    graph_draw(g, pos=g.vp["pos"], output_size=(1000, 1000), 
               edge_control_points=g.ep["cts"], edge_pen_width=g.ep["pen_width"],
               vertex_size=15, vertex_text=g.vp["text"], vcmap=matplotlib.cm.inferno,
               output=filename)


    
def animate_network(g, time_series, input_time_serie=None, output_time_serie=None):
    max_value=time_series.max()
    duration, n = len(time_series), len(time_series[0])
    filenames = []
    
    for i in tqdm(range(duration)):
        # vertex color
        fill_color = g.new_vp("vector<float>")
        for j in range(n):
            fill_color[j] = state_to_color(time_series[i][j], max_value)
        
        if input_time_serie is not None:
            fill_color[n] = state_to_color(input_time_serie[i], max_value)
            
        if output_time_serie is not None:
            fill_color[n+1 if input_time_serie is not None else n] = state_to_color(output_time_serie[i], max_value)
      
        output_filename= f"/tmp/frame_{i}.png"
        graph_draw(g, pos=g.vp["pos"], output_size=(1000, 1000), 
                   edge_control_points=g.ep["cts"], edge_pen_width=g.ep["pen_width"],
                   vertex_fill_color=fill_color, vertex_size=30, vertex_text=g.vp["text"], 
                   output=output_filename)
        filenames.append(output_filename)
            
    with imageio.get_writer('/tmp/network_animation.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
