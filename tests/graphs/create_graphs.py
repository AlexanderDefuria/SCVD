from src.graphml import NodeId, Abstraction, Mappings, apply_abstraction_mapping, build_abstraction_mapping, get_all_assignments, get_assignment_constituients, get_subgraph_from_edge_label

import pytest
import networkx as nx
import polars as pd
import os
from unittest.mock import patch, MagicMock
from pathlib import Path



class TestCreateGraphs():
    def test_create_graphs(self):
        

