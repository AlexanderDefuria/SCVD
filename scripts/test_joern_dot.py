from pathlib import Path
from typing import Iterable, List
from networkx.drawing.nx_agraph import read_dot, to_agraph
import os

from src.utils import interact

example_sc = """
int FF_ARRAY_ELEMS(int *arr) {
    return sizeof(arr) / sizeof(arr[0]);
}

float x = 0.0f;

int s337m_get_offset_and_codec(void *avctx, uint64_t state, int data_type, int data_size, int *offset, void *codec);

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
block_nodes = (
    "BLOCK",
    "CONTROL_STRUCTURE",
    "RETURN",
)

def select_global_graph_file(dir: Path) -> Path:
    """Select the global graph file from the given graph."""
    for file in os.listdir(dir):
        if file.endswith(".dot"):
            head = open(dir / file).readline().rstrip()
            # print(f"Processing file: {file} with head: {head}")
            if "global" in head:
                print(file)
                return dir/ file

    raise FileNotFoundError("No global graph file found in the directory.")


file_path = Path("data/.cache/test.c")
joern_cpg_bin = Path("data/.cache/test.cpg.bin")
dot_output = Path("data/.cache/out/")
name = "project_commit"

if file_path.exists():
    file_path.unlink()
if joern_cpg_bin.exists():
    joern_cpg_bin.unlink()
dot_output.mkdir(parents=True, exist_ok=True)

with open(file_path, "w") as f:
    f.write(example_sc)
    f.flush()
    f.close()

# call Joern to dump the CFG
VERBOSE = " > /dev/null 2>&1"
os.system(f"joern-parse {file_path} --output {joern_cpg_bin} {VERBOSE}")
for graph_type in ["ast", "cfg", "cpg14", "cdg", "ddg", "pdg"]:
    os.system(f"rm -rf {dot_output}/cfg {VERBOSE}")
    os.system(f"joern-export {joern_cpg_bin} --repr {graph_type} --out {dot_output}/{graph_type} {VERBOSE}")
    produced_dot_file: Path = select_global_graph_file(dot_output / graph_type)
    os.system(f"mv {produced_dot_file} {dot_output}/{graph_type}_{name}.dot {VERBOSE}")
    os.system(f"rm -rf {dot_output}/{graph_type}/ {VERBOSE}")

def get_declaration(graph, node, variable_dict) -> None:
    """Extract the identifier from the node."""
    var_name = node["label"].split(": ")[0].split(" ")[-1].replace("*", "").replace("&", "").strip()
    var_type = node["label"].split(": ")[1].strip()

    if "[" in var_name:
        var_name = var_name.split("[")[0].strip()
    if "[" in var_type:
        var_type = var_type.split("[")[0].strip() + "[]"

    node["type"] = var_type
    node["name"] = var_name
    variable_dict[var_name] = var_type


def get_parameter(param_node, variable_dict) -> None:
    if param_node["node_class"] == "PARAM":
        param_type = param_node["label"].split(" ")[0]
        param_name = param_node["label"].split(" ")[-1].replace("*", "").replace("&", "").strip()
        param_node["type"] = param_type
        param_node["name"] = param_name
        variable_dict[param_name] = param_type


def get_identifier(node) -> str:
    """Extract the identifier from the node."""
    return node["label"].split("<BR/>")[1]


def get_literal(node) -> str:
    return node["label"].split("<BR/>")[1]


def get_operator(node) -> str:
    return node["label"].split("<BR/>")[0].split(", ")[0]


def traverse_operator(graph, node, variable_dict) -> None:
    children_ids: List[str] = list(graph.neighbors(node["id"]))
    operator = get_operator(node)

    if len(children_ids) == 0:
        raise ValueError("No children found for assignment node")

    if len(children_ids) == 1:
        traverse_block(graph, node, variable_dict)
        child = graph.nodes[children_ids[0]]
        if "type" in child:
            node["type"] = child["type"]
        if "constant" in child:
            node["constant"] = child["constant"]

    elif len(children_ids) == 2:
        # Handle the case where there are two children
        # The first child is the variable being assigned to
        # The second child is the value being assigned
        traverse_block(graph, node, variable_dict)
        if "constant" in graph.nodes[children_ids[1]]:
            node["constant"] = graph.nodes[children_ids[1]]["constant"]
            if "type" in graph.nodes[children_ids[1]]:
                node["type"] = graph.nodes[children_ids[1]]["type"]
            if "type" in graph.nodes[children_ids[0]]:
                node["type"] = graph.nodes[children_ids[0]]["type"]

    if operator == "<operator>.cast":
        casted_type = node["label"].split("<BR/>")[1].split(")")[0].split("(")[1]
        node["type"] = casted_type
        print(f"casted_type: {casted_type}")


def traverse_block(graph, node, variable_dict):
    # Recursively traverse the graph to find the data type of a variable
    # Depth-first.
    children_ids: Iterable[str] = graph.neighbors(node["id"])
    children = [graph.nodes[child_id] for child_id in children_ids]

    for child in children:
        if child["node_class"] in ["LOCAL"]:
            get_declaration(graph, child, variable_dict)

        elif child["node_class"] in ["LITERAL"]:
            child["constant"] = get_literal(child)

        elif child["node_class"] in ["IDENTIFIER", "FIELD_IDENTIFIER"]:
            variable_name = get_identifier(child)
            node["type"] = variable_dict.get(variable_name, "unknown")

        elif "<operator>" in child["node_class"]:
            traverse_operator(graph, child, variable_dict)

        elif child["node_class"] in block_nodes:
            traverse_block(graph, child, variable_dict)

        elif child["node_class"] not in ["RETURN", "METHOD_RETURN"]:
            child["api"] = child["label"].split("<BR/>")[0]
            traverse_block(graph, child, variable_dict)


ast = read_dot(f"{dot_output}/ast_{name}.dot")
cfg = read_dot(f"{dot_output}/cfg_{name}.dot")


for node_id in ast.nodes():
    node = ast.nodes[node_id]
    label = node.get("label", "")
    label = label.replace("&lt;", "<")
    label = label.replace("&gt;", ">")
    if label == "":
        continue

    parts = label.split("<BR/>")
    node_class = parts[0].split(", ")[0]
    node_line = parts[0].split(", ")[1]
    operator = get_operator(node)

    node["label"] = label
    node["operator"] = operator
    node["node_class"] = node_class
    node["line"] = node_line
    node["id"] = node_id


root_nodes = (node for node, out_degree in ast.in_degree(ast.nodes) if out_degree == 0)
print(f"Root nodes in AST: {len(list(root_nodes))}")
print(list(root_nodes))
interact(locals=locals())
variable_dict = {}
for root_node in root_nodes:
    traverse_block(ast, root_node, variable_dict)
print(variable_dict)


for node_id in cfg.nodes():
    ast_node = ast.nodes[node_id]
    cfg_node = cfg.nodes[node_id]

    cfg_node["type"] = ast_node.get("type", "unknown")
    cfg_node["constant"] = ast_node.get("constant", " ")
    cfg_node["operator"] = ast_node.get("operator", " ")
    cfg_node["label"] = "\n".join([cfg_node["label"], cfg_node["type"], cfg_node["constant"], cfg_node["operator"]])

# Save to dot file
cfg.graph["label"] = "Control Flow Graph"

to_agraph(cfg).draw(f"{dot_output}/cfg_{name}.png", format="png", prog="dot", args="-Gdpi=500")
to_agraph(ast).draw(f"{dot_output}/ast_{name}.png", format="png", prog="dot", args="-Gdpi=500")
