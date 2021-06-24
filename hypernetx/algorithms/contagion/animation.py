from collections import defaultdict
import hypernetx as hnx
from celluloid import Camera

def contagion_animation(fig, H, transition_events, node_state_color_dict, edge_state_color_dict, node_radius=3, fps=1):
    nodeState = defaultdict(lambda: "S")

    camera = Camera(fig)

    for t in sorted(list(transition_events.keys())):
        edgeState = defaultdict(lambda: "OFF")

        # update edge and node states
        for event in transition_events[t]:
            status = event[0]
            node = event[1]

            # update node states
            nodeState[node] = status
        
            try:
                # update the edge transmitters list if they are neighbor-dependent transitions
                edgeID = event[2]
                if edgeID is not None:
                    edgeState[edgeID] = status
            except:
                pass

        kwargs = {'layout_kwargs': {'seed': 39}}

        nodeStyle = {'facecolors': [node_state_color_dict[nodeState[node]] for node in H.nodes]}
        edgeStyle = {'facecolors': [edge_state_color_dict[edgeState[edge]] for edge in H.edges], 'edgecolors':'black'}
        
        # draw hypergraph
        hnx.drawing.draw(H, node_radius=node_radius, nodes_kwargs=nodeStyle, edges_kwargs=edgeStyle, with_edge_labels=False, with_node_labels=False, **kwargs)
        camera.snap()

    return camera.animate(interval=1000/fps)