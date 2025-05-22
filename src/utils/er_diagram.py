from graphviz import Digraph

def render_er_diagram(schema_dict):
    dot = Digraph()
    for table, cols in schema_dict.items():
        col_str = "\n".join(cols)
        dot.node(table, f"{table}\n{col_str}", shape="box")
    return dot.source
