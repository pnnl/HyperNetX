.. _widget:


================
Hypernetx-Widget
================

.. image:: images/WidgetScreenShot.png
   :width: 300px
   :align: right

Overview
--------
The HyperNetXWidget_ is an addon for HNX, which extends the built in visualization 
capabilities of HNX to a JavaScript based interactive visualization. The tool has two main interfaces, 
the hypergraph visualization and the nodes & edges panel.
You may `demo the widget here <https://pnnl.github.io/hypernetx-widget/>`_

Installation
------------
The HypernetxWidget_ is available on `GitHub <https://github.com>`_ and may be
installed using pip:

    >>> pip install hnxwidget

Using the Tool
--------------

Layout
^^^^^^
The hypergraph visualization is an Euler diagram that shows nodes as circles and hyper edges as outlines 
containing the nodes/circles they contain. The visualization uses a force directed optimization to perform 
the layout. This algorithm is not perfect and sometimes gives results that the user might want to improve upon. 
The visualization allows the user to drag nodes and position them directly at any time. The algorithm will 
re-position any nodes that are not specified by the user. Ctrl (Windows) or Command (Mac) clicking a node 
will release a pinned node it to be re-positioned by the algorithm.

Selection
^^^^^^^^^
Nodes and edges can be selected by clicking them. Nodes and edges can be selected independently of each other, 
i.e., it is possible to select an edge without selecting the nodes it contains. Multiple nodes and edges can 
be selected, by holding down Shift while clicking. Shift clicking an already selected node will de-select it. 
Clicking the background will de-select all nodes and edges. Dragging a selected node will drag all selected 
nodes, keeping their relative placement.
Selected nodes can be hidden (having their appearance minimized) or removed completely from the visualization. 
Hiding a node or edge will not cause a change in the layout, wheras removing a node or edge will. 
The selection can also be expanded. Buttons in the toolbar allow for selecting all nodes contained within selected edges, 
and selecting all edges containing any selected nodes.
The toolbar also contains buttons to select all nodes (or edges), un-select all nodes (or edges), 
or reverse the selected nodes (or edges). An advanced user might:

* **Select all nodes not in an edge** by: select an edge, select all nodes in that edge, then reverse the selected nodes to select every node not in that edge.
* **Traverse the graph** by: selecting a start node, then alternating select all edges containing selected nodes and selecting all nodes within selected edges
* **Pin Everything** by: hitting the button to select all nodes, then drag any node slightly to activate the pinning for all nodes.
  
Side Panel
^^^^^^^^^^
Details on nodes and edges are visible in the side panel. For both nodes and edges, a table shows the node name, degree (or size for edges), its selection state, removed state, and color. These properties can also be controlled directly from this panel. The color of nodes and edges can be set in bulk here as well, for example, coloring by degree.

Other Features
^^^^^^^^^^^^^^
Nodes with identical edge membership can be collapsed into a super node, which can be helpful for larger hypergraphs. Dragging any node in a super node will drag the entire super node. This feature is available as a toggle in the nodes panel.

The hypergraph can also be visualized as a bipartite graph (similar to a traditional node-link diagram). Toggling this feature will preserve the locations of the nodes between the bipartite and the Euler diagrams.

.. _HypernetxWidget: https://github.com/pnnl/hypernetx-widget
