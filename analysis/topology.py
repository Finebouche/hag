from graph_tool.all import arf_layout, Graph, graph_draw, get_hierarchy_tree, radial_tree_layout, \
    get_hierarchy_control_points, minimize_nested_blockmodel_dl
import matplotlib
import numpy as np


def create_network(W, Win=None, Wout=None):
    # Size of the number of connection in the connection matrice
    row, col = W.row, W.col
    W = W.toarray() / np.amax(W)
    n = len(W)

    # We add the edges
    edge_index = []
    pen_width = []
    for i in range(len(col)):
        edge_index.append((row[i], col[i]))
        pen_width.append(W[row[i]][col[i]])

    if Win is not None:
        Win = Win.toarray() / np.amax(Win)
        for i in range(n):
            edge_index.append((n, i))
            pen_width.append(Win[i])

    if Wout is not None:
        Wout = Wout.toarray() / np.amax(Wout)
        for i in range(n):
            edge_index.append((i, n + 1))
            pen_width.append(Wout[i])

    edge_index = np.array(edge_index)
    g = Graph(directed=True)
    g.add_edge_list(edge_index)
    edge_pen_width = g.new_edge_property("double")
    edge_pen_width.a = np.abs(pen_width) * 4
    g.properties[("e", "edge_pen_weight")] = edge_pen_width

    return g


def draw_network(g, n):
    vertex_text = []
    text = g.new_vp("double")
    for v in g.vertex_index:
        vertex_text.append(v)
    text.a = vertex_text
    pos = arf_layout(g)
    # Position of the output node
    pos[n + 1] = [7, 4]
    # Position of the input node
    pos[n] = [1, 4]

    # The curvatures of the edges
    state = minimize_nested_blockmodel_dl(g)
    t = get_hierarchy_tree(state)[0]
    tpos = radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
    cts = get_hierarchy_control_points(g, t, tpos)

    graph_draw(g, pos=pos, output_size=(1000, 1000), edge_control_points=cts,
               edge_pen_width=g.ep["edge_pen_weight"], vertex_size=10, vertex_text=text,
               vcmap=matplotlib.cm.inferno)