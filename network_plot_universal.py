import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

# Set a default font that works across systems
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Bitstream Vera Sans", "sans-serif"]

# Create graph object
G = nx.Graph()

# Add nodes
nodes = ["javascript", "Matplotlib", "OpenGL", "d3js", "Vulkan", "cufflinks", "Lighting", "ipyvolume", 
         "ipyleaflet", "plotly", "toyplot", "bokeh", "bqplot", "pythreejs", "vispy", "glumpy", "pyglet", 
         "GR Framework", "VTK", "Fury", "PyVista", "mayavi", "galry", "visvis", "holoviews", "datashader", 
         "Vaex", "mpl3d", "d3po", "vincent", "vega", "altair", "vega-lite", "cartopy", "YT", "basemap", 
         "graph-tool", "graphviz", "networkx", "yellowbrick", "scikit-plot", "ggpy", "plotnine", "seaborn", 
         "glueviz", "pandas", "pygal", "chaco", "PyQTGraph", "Datoviz"]
G.add_nodes_from(nodes)

# Add edges
edges = [
    ("javascript", "cufflinks"), ("javascript", "Lighting"), ("javascript", "ipyvolume"),
    ("javascript", "ipyleaflet"), ("javascript", "plotly"), ("javascript", "toyplot"),
    ("javascript", "bokeh"), ("javascript", "bqplot"), ("javascript", "pythreejs"),
    ("Matplotlib", "cartopy"), ("Matplotlib", "YT"), ("Matplotlib", "basemap"),
    ("Matplotlib", "graph-tool"), ("Matplotlib", "graphviz"), ("Matplotlib", "networkx"),
    ("Matplotlib", "yellowbrick"), ("Matplotlib", "scikit-plot"), ("Matplotlib", "ggpy"),
    ("Matplotlib", "plotnine"), ("Matplotlib", "seaborn"), ("Matplotlib", "glueviz"),
    ("Matplotlib", "pandas"), ("OpenGL", "vispy"), ("OpenGL", "glumpy"), ("OpenGL", "pyglet"),
    ("OpenGL", "GR Framework"), ("OpenGL", "VTK"), ("OpenGL", "visvis"), ("OpenGL", "galry"),
    ("d3js", "mpl3d"), ("d3js", "d3po"), ("d3js", "vincent"), ("d3js", "vega"),
    ("d3js", "altair"), ("d3js", "vega-lite"), ("Vulkan", "Datoviz"),
    ("javascript", "Matplotlib"), ("javascript", "OpenGL"), ("javascript", "d3js"),
    ("javascript", "Vulkan"), ("OpenGL", "Fury"), ("OpenGL", "PyVista"), ("OpenGL", "mayavi"),
    ("Matplotlib", "holoviews"), ("Matplotlib", "datashader"), ("Matplotlib", "Vaex"),
    ("javascript", "pygal"), ("Matplotlib", "chaco"), ("Matplotlib", "PyQTGraph")
]
G.add_edges_from(edges)

# Set node colors
node_color_dict = {
    "javascript": "#FF6B6B",    # Red
    "Matplotlib": "#4ECDC4",    # Cyan
    "OpenGL": "#45B7D1",        # Blue
    "d3js": "#FFA07A",          # Light orange
    "Vulkan": "#98D8C8"         # Light green
}
node_colors = [node_color_dict.get(node, "#E0E0E0") for node in G.nodes()]  # Gray for others

# Set node sizes
node_size = [3000 if node in node_color_dict else 1000 for node in G.nodes()]

# Draw the graph
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=0.3, iterations=50)  # Layout parameters

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.8)

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, alpha=0.5, edge_color="#888888")

# Draw labels using the default font setting
nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

# Remove axis
plt.axis("off")

# Add title using the default font setting
plt.title("Visualization Library Relationship Network", fontsize=16)

# Show plot
plt.tight_layout()
plt.show()
    