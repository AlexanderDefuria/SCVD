from src.graphml import NodeId, Abstraction, Mappings, apply_abstraction_mapping, build_abstraction_mapping, get_all_assignments, get_assignment_constituients, get_subgraph_from_edge_label

import pytest
import networkx as nx
import polars as pd
import os
from unittest.mock import patch, MagicMock
from pathlib import Path


"""
Example Node-Abstraction Dictionary:
"1": {
    data_type: "int",
    api_call: "s337m_get_offset_and_codec",
    constant: "0",
    operator: "*",
},
"2": {
    data_type: "int*",
    api_call: "malloc",
    constant: "4",
    operator: None,
},
"3": {
    data_type: "int",
    api_call: "AV_RL16",
    constant: None,
    operator: None,
},
"4": {
    data_type: "int",
    api_call: "AV_RL16",
    constant: "2",
    operator: "+",
},
"""


class TestGetAllAssignments:
    def test_get_all_assignments(self):
        pass
    
    def test_get_all_assignments_empty_graph(self):
        G = nx.MultiDiGraph()
        result = get_all_assignments(G)
        assert result == []

    def test_get_all_assignments_single_node(self):
        G = nx.MultiDiGraph()
        G.add_node("1", METHOD_FULL_NAME="<operator>.assignment")
        G.add_node("2", METHOD_FULL_NAME="<operator>.constant")
        G.add_node("3", METHOD_FULL_NAME="test()")
        result = get_all_assignments(G)
        assert result == ['1']



