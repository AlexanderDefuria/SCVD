from pathlib import Path
from typing import Any, Dict, Iterable, List, TypeAlias
from networkx.readwrite import read_graphml
from networkx import MultiDiGraph
from polars import DataFrame
import networkx as nx
import os

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


EXAMPLE_SC = """
#define TEST 1

int FF_ARRAY_ELEMS(int *arr) {
    return sizeof(arr) / sizeof(arr[0]);
}

float x = 0.0f;

int s337m_get_offset_and_codec(void *avctx, uint64_t state, int data_type, int data_size, int *offset, void *codec);

char *str = "Hello, World!";
char *str_two_elctric_boogaloo = NULL;
int check = TEST;

static int s337m_probe(AVProbeData *p)
{
    uint64_t state = 0;
    int markers[3] = { 0 };
    int i, pos, sum, max, data_type, data_size, offset;
    uint8_t *buf;
    int *test = (int*) malloc(4);
    int a_test = 1;
    float b_test = (float) a_test;

    for (pos = 0; pos < p->buf_size; pos++) {
        state = (state << 8) | p->buf[pos];
        if (!IS_LE_MARKER(state))
            continue;

        buf = p->buf + pos + 1;
        if (IS_16LE_MARKER(state)) {
            data_type = AV_RL16(buf    );
            data_size = AV_RL16(buf + 2);
        } else {
            data_type = AV_RL24(buf    );
            data_size = AV_RL24(buf + 3);
        }

        if (s337m_get_offset_and_codec(NULL, state, data_type, data_size, &offset, NULL))
            continue;

        i = IS_16LE_MARKER(state) ? 0 : IS_20LE_MARKER(state) ? 1 : 2;
        markers[i]++;

        pos  += IS_16LE_MARKER(state) ? 4 : 6;
        pos  += offset;
        state = 0;
    }

    sum = max = 0;
    for (i = 0; i < FF_ARRAY_ELEMS(markers); i++) {
        sum += markers[i];
        if (markers[max] < markers[i])
            max = i;
    }

    if (markers[max] > 3 && markers[max] * 4 > sum * 3)
        return AVPROBE_SCORE_EXTENSION + 1;

    return 0;
}
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

def get_subgraph_from_edge_label(graph: MultiDiGraph, edge_label: str) -> MultiDiGraph:
    """
    Get a subgraph from the graph based on the edge label.
    :param graph: The input graph.
    :param edge_label: The label of the edges to filter by.
    :return: A subgraph containing only the edges with the specified label.
    """
    edges = [(u, v, k) for u, v, k, d in graph.edges(data=True, keys=True) if d.get("labelE") == edge_label]
    subgraph: MultiDiGraph = graph.edge_subgraph(edges)  # type: ignore
    return subgraph


def get_all_assignments(graph: MultiDiGraph) -> List:
    """
    Get all assignment nodes from the graph
    """
    assignments = []
    for node, data in graph.nodes(data=True):
        if data.get("METHOD_FULL_NAME") in all_assignment_types:
            assignments.append(node)
    return assignments


def get_assignment_constituients(graph: MultiDiGraph, node: str) -> Abstraction:
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


def build_abstraction_mapping(abstraction_data: Dict[NodeId, Abstraction]) -> Mappings:
    """
    Build a mapping of unique values for each column in the abstraction data based on their frequency.
    """
    assignments_df = DataFrame(
        list(abstraction_data.values()),
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


def parse_joern_cpg(file_path: Path) -> MultiDiGraph:
    """
    Parse the Joern CPG file and return a MultiDiGraph.
    :param file_path: The path to the Joern CPG file.
    :return: A MultiDiGraph representing the CPG.
    """
    VERBOSE = " > /dev/null 2>&1"
    os.system(f"joern-parse {file_path} --output {joern_cpg_bin} {VERBOSE}")
    os.system(f"rm -rf {output_path}")
    os.system(f"joern-export {joern_cpg_bin} --repr=all --format=graphml --out {output_path}{VERBOSE}")
    return read_graphml(f"{output_path}/export.xml")


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    assert root_dir.exists(), "Root directory does not exist"
    assert root_dir.name == "SCVD", "Root directory name should be 'SCVD'"

    cache_dir = root_dir / Path("data/.cache")
    file_path = root_dir / Path("data/.cache/test.c")
    joern_cpg_bin = root_dir / Path("data/.cache/test.cpg.bin")
    output_path = root_dir / Path("data/.cache/out/")
    name = "project_commit"
    
    os.makedirs(cache_dir, exist_ok=True)
    if file_path.exists():
        file_path.unlink()
    if joern_cpg_bin.exists():
        joern_cpg_bin.unlink()
    
    with open(file_path, "w") as f:
        f.write(EXAMPLE_SC)
        f.flush()
        f.close()
    
    cpg = parse_joern_cpg(file_path)

    # Get CFG and AST from the graph based on the edge labels
    ast: MultiDiGraph = get_subgraph_from_edge_label(cpg, "AST")
    cfg: MultiDiGraph = get_subgraph_from_edge_label(cpg, "CFG")
    
    # Get all assignment nodes from the CFG
    assignment_nodes = get_all_assignments(cfg)
    assignment_node_data = [cfg.nodes[node] for node in assignment_nodes]
    abstraction_data: Dict[NodeId, Abstraction] = {}
    
    for node, data in zip(assignment_nodes, assignment_node_data):
        # descendant_nodes: List[NodeId] = list(nx.descendants_at_distance(ast, node, 1))
        # descendant_nodes.sort(key=lambda x: ast.nodes[x].get("ARGUMENT_INDEX", 0))  # Sort by argument number (1 is the first argument)
        abstraction_data[node] = get_assignment_constituients(ast, node)
        # print(f"Node: {node}, Line: {data.get('LINE_NUMBER')}, Code: {data.get('CODE', 'N/A')}")
        # print(f"Dict for node {node}: {abstraction_data[node]}\n\n")
    
    # for node, abstraction in abstraction_data.items():
    # print(f"{cfg.nodes[node].get("CODE", 'N/A')} --> {abstraction}")
    
    mappings: Mappings = build_abstraction_mapping(abstraction_data)
    for node in cfg.nodes:
        # If node is in abstraction_data, replace the node data add the abstraction data as "ABSTRACTION"
        # Otherwise set "ABSTRACTION" to default dict
        if node in abstraction_data:
            cfg.nodes[node]["ABSTRACTION"] = apply_abstraction_mapping(mappings, abstraction_data[node])
            print(f"Node: {node}, Abstraction: {cfg.nodes[node]['ABSTRACTION']}")
        else:
            # Non-assignment nodes will have an empty abstraction
            cfg.nodes[node]["ABSTRACTION"] = {
                "constant": 1,
                "api_call": 1,
                "data_type": 1,
                "operator": 1,
            }
    
    
