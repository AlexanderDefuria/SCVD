from code import interact
from pathlib import Path
from re import M
from typing import Any, Dict, Iterable, List, Optional, TypeAlias
from networkx.readwrite import graphml, read_graphml
from networkx import DiGraph, MultiDiGraph, write_graphml
from polars import DataFrame
import networkx as nx
import pickle
import json
import os

from torch_geometric.data.data import BaseData
from torch_geometric.utils import from_networkx

from src import *

NodeId: TypeAlias = str
Abstraction: TypeAlias = Dict[str, str]
Mappings: TypeAlias = Dict[str, Dict[str, int]]

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

traditional_assignment = (
    "<operator>.assignmentDivision",
    "<operator>.assignmentExponentiation",
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentModulo",
    "<operator>.assignmentMultiplication",
    "<operator>.assignment",
    "<operator>.assignmentOr",
    "<operator>.assignmentAnd",
    "<operator>.assignmentXor",
    "<operator>.assignmentArithmeticShiftRight",
    "<operator>.assignmentLogicalShiftRight",
    "<operator>.assignmentShiftLeft",
)

increment_assignment = (
    "<operator>.preIncrement",
    "<operator>.preDecrement",
    "<operator>.postIncrement",
    "<operator>.postDecrement",
)

all_assignment_types = traditional_assignment + increment_assignment


def get_subgraph_from_edge_label(graph: MultiDiGraph, edge_label: str) -> DiGraph:
    """
    Get a subgraph from the graph based on the edge label.
    :param graph: The input graph.
    :param edge_label: The label of the edges to filter by.
    :return: A subgraph containing only the edges with the specified label.
    """
    edges = [(u, v, k) for u, v, k, d in graph.edges(data=True, keys=True) if d.get("labelE") == edge_label]
    subgraph: DiGraph = graph.edge_subgraph(edges)  # type: ignore
    return subgraph


def get_all_assignments(graph: DiGraph) -> List:
    """
    Get all assignment nodes from the graph
    """
    assignments = []
    for node, data in graph.nodes(data=True):
        if data.get("METHOD_FULL_NAME") in all_assignment_types:
            assignments.append(node)
    return assignments


def get_assignment_constituients(graph: DiGraph, node: str) -> Abstraction:
    """
    BFS traversal to find the first constant, api_call, data_type, and operator based on the node descendants
    :param graph: The input graph.
    :param node: The node to start the BFS traversal from.
    :return: A dictionary containing the constant, api_call, data_type, and operator.
    """
    constant, api_call, data_type, operator = None, None, None, None
    for i, descendant in enumerate(nx.bfs_tree(graph, node)):
        if i == 0:
            # If the statement is a single line and is incrementing. Set the operator to the method name
            if graph.nodes[descendant].get("METHOD_FULL_NAME") in increment_assignment:
                operator = graph.nodes[descendant]["METHOD_FULL_NAME"]
            continue  # Skip the first node (the assignment itself)

        # If we start to get noise from the descendants, we can break early
        if i > 10:  # TODO do we delete this?
            break

        # @ Uncomment the following line to debug the descendants
        # print(f"Descendant {i}: {descendant}, Data: {graph.nodes[descendant]}")

        # Extract the LITERAL
        if constant is None and graph.nodes[descendant].get("labelV") == "LITERAL":
            constant = graph.nodes[descendant]["CODE"]

        # Extract the API call
        if data_type is None and graph.nodes[descendant].get("TYPE_FULL_NAME") is not None:
            data_type = graph.nodes[descendant]["TYPE_FULL_NAME"]

        # Extract the API call and operator
        # These use the same METHOD_FULL_NAME field but all operators start with "<operator>."
        # We can differentiate them based on this prefix.
        if graph.nodes[descendant].get("METHOD_FULL_NAME") is not None:
            if api_call is None and not graph.nodes[descendant]["METHOD_FULL_NAME"].startswith("<operator>."):
                api_call = graph.nodes[descendant]["METHOD_FULL_NAME"]
            elif operator is None and graph.nodes[descendant]["METHOD_FULL_NAME"].startswith("<operator>."):
                operator = graph.nodes[descendant]["METHOD_FULL_NAME"]

        # Break the loop if all constituents are found
        if constant and api_call and data_type and operator:
            break

    constant = constant if constant is not None else ""
    api_call = api_call if api_call is not None else ""
    data_type = data_type if data_type is not None else ""
    operator = operator if operator is not None else ""

    assert isinstance(constant, str), "Constant should be a string or None"
    assert isinstance(api_call, str), "API call should be a string or None"
    assert isinstance(data_type, str), "Data type should be a string or None"
    assert isinstance(operator, str), "Operator should be a string or None"

    return {
        "constant": constant,
        "api_call": api_call,
        "data_type": data_type,
        "operator": operator,
    }


def get_abstraction_dictionary(cfg_with_assignments: DiGraph) -> Dict[NodeId, Abstraction]:
    abstraction_data: Dict[NodeId, Abstraction] = {}
    for node, data in cfg_with_assignments.nodes(data=True):
        if "ABSTRACTION" in data:
            abstraction_data[node] = data["ABSTRACTION"]
    return abstraction_data


def build_abstraction_mapping(abstraction_dict) -> Mappings:
    """
    Build a mapping of unique values for each column in the abstraction data based on their frequency.
    """
    assignments_df = DataFrame(
        list(abstraction_dict.values()),
        schema=[
            ("constant", str),
            ("api_call", str),
            ("data_type", str),
            ("operator", str),
        ],
    )

    # Count each unique value in the each column
    mappings: Mappings = {}
    for column in assignments_df.columns:
        value_counts = assignments_df[column].value_counts()
        value_counts = value_counts.sort("count", descending=True)
        mappings[column] = {row[column]: row["index"] for row in value_counts.with_row_index("index").iter_rows(named=True)}

    # For all non-zero values, increment the value by 1
    # TODO Confirm the correct way of handling non-assignment values is by assigning a 1 for "unknown".
    for column in mappings:
        for key in mappings[column]:
            mappings[column][key] += 1 if mappings[column][key] > 0 else 0

    return mappings


def apply_abstraction_mapping(mappings: Mappings, abstraction: Abstraction) -> Dict[str, int]:
    """
    Apply the abstraction mapping to the given abstraction.
    :param mappings: The mappings to apply.
    :param abstraction: The abstraction to apply the mappings to.
    :return: A dictionary with the mapped values.
    """
    return {key: mappings[key].get(abstraction[key], 0) for key in abstraction.keys()}


def parse_joern_cpg(file_path: Path, project_commit: str) -> MultiDiGraph:
    """
    Parse the Joern CPG file and return a MultiDiGraph.
    :param file_path: The path to the Joern CPG file.
    :return: A MultiDiGraph representing the CPG.
    """
    cpg_bin_path = JOERN_OUTPUT_CACHE / f"{project_commit}.cpg.bin"
    export_dir = JOERN_OUTPUT_CACHE / project_commit
    graphml_path = export_dir / "export.xml"
    os.system(f"rm -rf {export_dir}")  # Clear the previous output for joern
    os.system(f"joern-parse {file_path} --output {cpg_bin_path} {VERBOSE}")
    os.system(f"joern-export {cpg_bin_path} --repr=all --format=graphml --out {export_dir}{VERBOSE}")
    return read_graphml(f"{graphml_path}")


def extract_assignments(cpg: MultiDiGraph) -> DiGraph:
    """
    Extract assignments from the CPG and return a CFG with assignments.
    :param cpg: The CPG to extract assignments from.
    :return: A DiGraph representing the CFG with assignments.
    """
    # Get CFG and AST from the graph based on the edge labels
    ast: DiGraph = get_subgraph_from_edge_label(cpg, "AST")
    cfg: DiGraph = get_subgraph_from_edge_label(cpg, "CFG")

    # Get all assignment nodes from the CFG
    assignment_nodes = get_all_assignments(cfg)
    for node in assignment_nodes:
        cfg.nodes[node]["ABSTRACTION"] = get_assignment_constituients(ast, node)

    return cfg


def save_cfg_assignments(file_name: str, cfg: DiGraph) -> None:
    """
    Save the CFG to a file.
    :param file_path: The path to the file where the graph will be saved.
    :param cfg: The CFG to save.
    """
    file_path = JOERN_INTERMIDATE_GRAPH / Path(file_name)
    assert str(file_path).endswith(".graphml"), f"File path must have a .graphml suffix..."

    for node in cfg.nodes:
        abstraction = cfg.nodes[node].get("ABSTRACTION")
        if abstraction is not None:
            cfg.nodes[node]["ABSTRACTION"] = json.dumps(abstraction)

    write_graphml(cfg, file_path)


def load_assignment_cfg(file_name: str) -> DiGraph:
    """
    Load a saved graph from a file.
    :param file_path: The path to the saved graph file.
    :return: A dictionary containing the CFG.
    """
    file_path = JOERN_INTERMIDATE_GRAPH / Path(file_name)
    assert str(file_path).endswith(".graphml"), f"File path must have a .graphml suffix..."
    assert file_path.exists(), "File does not exist"
    output: DiGraph = read_graphml(file_path)

    count = 0
    for node in output.nodes:
        abstraction = output.nodes[node].get("ABSTRACTION")
        if abstraction is not None:
            count += 1
            output.nodes[node]["ABSTRACTION"] = json.loads(abstraction)
            # TODO potentially find a way to handle serde errors here.

    if count == 0:
        raise Exception(
            f"{file_name}: No assignments found in the graph. Abstraction mapping will not be applied. You are likely trying to run this on a graph that has not had the assignments extracted yet"
        )
    assert isinstance(output, DiGraph), f"Output should be a DiGraph now after serde: {type(output)}"
    return output


def apply_abstraction_mapping_to_graph(cfg: DiGraph, mappings: Mappings) -> None:
    for node in cfg.nodes:
        if cfg.nodes[node].get("ABSTRACTION") is not None:
            cfg.nodes[node]["ABSTRACTION"] = apply_abstraction_mapping(mappings, cfg.nodes[node]["ABSTRACTION"])
            print(f"Node: {node}, Abstraction: {cfg.nodes[node]['ABSTRACTION']}")
        else:
            # Non-assignment nodes will have an empty abstractionrwith a constant value of 1
            cfg.nodes[node]["ABSTRACTION"] = {
                "constant": 1,
                "api_call": 1,
                "data_type": 1,
                "operator": 1,
            }


def cfg_to_pyg_data(cfg: DiGraph) -> BaseData:
    for node in cfg.nodes:
        abstraction = cfg.nodes[node].get("ABSTRACTION")
        # Remove all attributes that are not part of the abstraction
        for key in list(cfg.nodes[node].keys()):
            if not key.startswith("ABSTRACTION"):
                del cfg.nodes[node][key]

        cfg.nodes[node]["ABSTRACTION"] = (
            abstraction
            if abstraction is not None
            else {
                "constant": None,
                "api_call": None,
                "data_type": None,
                "operator": None,
            }
        )

    return from_networkx(cfg)


def write_code_to_temp_file(data: Dict[str, str], project_commit: str) -> Path:
    """
    Write data to a temporary file and return the file path.
    :param data: The data to write to the file.
    :param file_name: The name of the file to write to.
    :return: The path to the temporary file.
    """
    # TODO CURRENTLY WE ARE APPENDING THE CODE SNIPPETS INTO A SINGLE FILE.
    # TODO WE MAY WANT TO CHANGE THIS IN THE FUTURE TO HAVE A SEPARATE FILE FOR EACH SOURCE FILE.
    temp_file_path = JOERN_OUTPUT_TEMP / f"{project_commit}_code.c"
    with open(temp_file_path, "w") as temp_file:
        for key, value in data.items():
            temp_file.write(f"\n\n\n//--------{key}-------\n {value}\n")
            temp_file.flush()
        temp_file.close()
    return temp_file_path


def run_graphml(code_snippet: str) -> None:
    """
    Run the graphml process with the given code snippet.
    THIS IS A TEST FUNCTION THAT WILL BE USED TO TEST THE GRAPHML PROCESSING.
    Reminder to use the functions contained within this as appropriate in the datasets.

    :param code_snippet: The code to test the process with.
    """

    assert ROOT_DIR.exists(), "Root directory does not exist"
    assert ROOT_DIR.name == "SCVD", "Root directory name should be 'SCVD'"

    project_commit = "sha_id"
    code_snippets = {"test.c": code_snippet}
    graph_name = f"{project_commit}.graphml"
    file_path = write_code_to_temp_file(data=code_snippets, project_commit=project_commit)

    # Creating the graph and getting all assignments
    cpg: MultiDiGraph = parse_joern_cpg(file_path, project_commit)
    cfg = extract_assignments(cpg)
    save_cfg_assignments(graph_name, cfg)

    # Building the abstraction mapping and applying it to the graph
    cfg: DiGraph = load_assignment_cfg(graph_name)
    abstraction_dict: Dict[NodeId, Abstraction] = get_abstraction_dictionary(cfg)
    mappings: Mappings = build_abstraction_mapping(abstraction_dict)
    apply_abstraction_mapping_to_graph(cfg, mappings)


if __name__ == "__main__":
    run_graphml(EXAMPLE_SC)
