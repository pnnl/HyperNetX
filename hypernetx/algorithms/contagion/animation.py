from collections import defaultdict
import hypernetx as hnx
from celluloid import Camera


def contagion_animation(
    fig,
    H,
    transition_events,
    node_state_color_dict,
    edge_state_color_dict,
    node_radius=1,
    fps=1,
):
    """
    A function to animate discrete-time contagion models for hypergraphs. Currently only supports a circular layout.

    Parameters
    ----------
    fig : matplotlib Figure object
    H : HyperNetX Hypergraph object
    transition_events : dictionary
        The dictionary that is output from the discrete_SIS and discrete_SIR functions with return_full_data=True
    node_state_color_dict : dictionary
        Dictionary which specifies the colors of each node state. All node states must be specified.
    edge_state_color_dict : dictionary
        Dictionary with keys that are edge states and values which specify the colors of each edge state
        (can specify an alpha parameter). All edge-dependent transition states must be specified
        (most common is "I") and there must be a a default "OFF" setting.
    node_radius : float, default: 1
        The radius of the nodes to draw
    fps : int > 0, default: 1
        Frames per second of the animation

    Returns
    -------
    matplotlib Animation object

    Notes
    -----

    Example::

        >>> import hypernetx.algorithms.contagion as contagion
        >>> import random
        >>> import hypernetx as hnx
        >>> import matplotlib.pyplot as plt
        >>> from IPython.display import HTML
        >>> n = 1000
        >>> m = 10000
        >>> hyperedgeList = [random.sample(range(n), k=random.choice([2,3])) for i in range(m)]
        >>> H = hnx.Hypergraph(hyperedgeList)
        >>> tau = {2:0.1, 3:0.1}
        >>> gamma = 0.1
        >>> tmax = 100
        >>> dt = 0.1
        >>> transition_events = contagion.discrete_SIS(H, tau, gamma, rho=0.1, tmin=0, tmax=tmax, dt=dt, return_full_data=True)
        >>> node_state_color_dict = {"S":"green", "I":"red", "R":"blue"}
        >>> edge_state_color_dict = {"S":(0, 1, 0, 0.3), "I":(1, 0, 0, 0.3), "R":(0, 0, 1, 0.3), "OFF": (1, 1, 1, 0)}
        >>> fps = 1
        >>> fig = plt.figure()
        >>> animation = contagion.contagion_animation(fig, H, transition_events, node_state_color_dict, edge_state_color_dict, node_radius=1, fps=fps)
        >>> HTML(animation.to_jshtml())
    """

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

        kwargs = {"layout_kwargs": {"seed": 39}}

        nodeStyle = {
            "facecolors": [node_state_color_dict[nodeState[node]] for node in H.nodes]
        }
        edgeStyle = {
            "facecolors": [edge_state_color_dict[edgeState[edge]] for edge in H.edges],
            "edgecolors": "black",
        }

        # draw hypergraph
        hnx.draw(
            H,
            node_radius=node_radius,
            nodes_kwargs=nodeStyle,
            edges_kwargs=edgeStyle,
            with_edge_labels=False,
            with_node_labels=False,
            **kwargs
        )
        camera.snap()

    return camera.animate(interval=1000 / fps)
