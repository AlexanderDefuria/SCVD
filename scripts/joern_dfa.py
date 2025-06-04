"""
Extract abstract dataflow features from graphs
Yields:
- storage/cache/bigvul/abstract_dataflow.csv (cache)
- storage/processed/bigvul/abstract_dataflow_hash_api_datatype_literal_operator.csv
"""

import argparse
import functools
import json
import re
import traceback
from multiprocessing import Pool

import networkx as nx
import pandas as pd
import code_gnn.analysis.dataflow as dataflow
import sastvd.helpers.datasets as svdds
import sastvd as svd
import tqdm

# Extract dataflow features from CPG

all_assignment_types = (
    "<operator>.assignmentDivision",
    "<operator>.assignmentExponentiation",
    "<operator>.assignmentPlus",
    "<operator>.assignmentMinus",
    "<operator>.assignmentModulo",
    "<operator>.assignmentMultiplication",
    "<operator>.preIncrement",
    "<operator>.preDecrement",
    "<operator>.postIncrement",
    "<operator>.postDecrement",
    "<operator>.assignment",
    "<operator>.assignmentOr",
    "<operator>.assignmentAnd",
    "<operator>.assignmentXor",
    "<operator>.assignmentArithmeticShiftRight",
    "<operator>.assignmentLogicalShiftRight",
    "<operator>.assignmentShiftLeft",
)

def is_decl(n_attr):
    # NOTE: this is local variable declarationsm
    # which are not considered definitions in formal DFA setting.
    # if n_attr["_label"] in ("LOCAL",):
    #     return True

    # https://github.com/joernio/joern/blob/15e241d3174ecba9e977a399793c9c6a1249d819/semanticcpg/src/main/scala/io/shiftleft/semanticcpg/language/operatorextension/package.scala
    return n_attr["_label"] == "CALL" and n_attr["name"] in all_assignment_types


def get_dataflow_features(ast, raise_all=False, verbose=False):
    try:
        ast = dataflow.sub(cpg, "AST")
        arg_graph = dataflow.sub(cpg, "ARGUMENT")
        labels = nx.get_node_attributes(cpg, "_label")
        code = nx.get_node_attributes(cpg, "code")
        names = nx.get_node_attributes(cpg, "name")

        def recurse_datatype(v):
            v_attr = cpg.nodes[v]
            if verbose:
                print("recursing", v, v_attr)

            name_idx = {
                "<operator>.indirectIndexAccess": 1,
                "<operator>.indirectFieldAccess": 1,
                "<operator>.indirection": 1,
                "<operator>.fieldAccess": 1,
                "<operator>.postIncrement": 1,
                "<operator>.postDecrement": 1,
                "<operator>.preIncrement": 1,
                "<operator>.preDecrement": 1,
                "<operator>.addressOf": 1,
                "<operator>.cast": 2,
                "<operator>.addition": 1,
            }
            if v_attr["_label"] == "IDENTIFIER":
                return v, v_attr["typeFullName"]
            elif v_attr["_label"] == "CALL":
                if v_attr["name"] in name_idx.keys():
                    # TODO: Get field data type, not struct data type
                    args = {cpg.nodes[s]["order"]: s for s in arg_graph.successors(v)}
                    arg = args[name_idx[v_attr["name"]]]
                    arg_attr = cpg.nodes[arg]
                    if verbose:
                        print("index", arg, arg_attr)
                        if v_attr["name"] == "<operator>.addition":
                            print("addition debug", v, v_attr, arg, arg_attr)
                    if arg_attr["_label"] == "IDENTIFIER":
                        return arg, arg_attr["typeFullName"]
                    elif arg_attr["_label"] == "CALL":
                        return recurse_datatype(arg)
                    else:
                        raise NotImplementedError(
                            f"recurse_datatype index could not handle {v} {v_attr} -> {arg} {arg_attr}"
                        )
            raise NotImplementedError(
                f"recurse_datatype var could not handle {v} {v_attr}"
            )

        def get_raw_datatype(decl):
            decl_attr = cpg.nodes[decl]

            if verbose:
                print("parent", decl, decl_attr)

            if decl_attr["_label"] == "LOCAL":
                return decl, decl_attr["typeFullName"]
            elif decl_attr["_label"] == "CALL" and decl_attr[
                "name"
            ] in all_assignment_types + ("<operator>.cast",):
                args = {cpg.nodes[s]["order"]: s for s in arg_graph.successors(decl)}
                return recurse_datatype(args[1])
            else:
                raise NotImplementedError(
                    f"""get_raw_datatype did not handle {decl} {decl_attr}"""
                )

        def grab_declfeats(node_id):
            fields = []
            try:
                ret = get_raw_datatype(node_id)
                if ret is not None:
                    child_id, child_datatype = ret
                    fields.append(("datatype", child_id, child_datatype))

                # create a copy of the AST with method definitions excluded.
                # this avoids an issue where some variable definitions descend to
                # method definitions (probably by mistake), shown in graph 3.
                my_ast = ast.copy()
                my_ast.remove_nodes_from(
                    [
                        n
                        for n, attr in ast.nodes(data=True)
                        if attr["_label"] == "METHOD"
                    ]
                )

                to_search = nx.descendants(my_ast, node_id)
                for n in to_search:
                    if verbose:
                        print(
                            f"{node_id} desc {n} {code.get(n, None)} {names.get(n, None)} {nx.shortest_path(ast, node_id, n)}"
                        )
                    if labels[n] == "LITERAL":
                        fields.append(("literal", n, code.get(n, pd.NA)))
                    if labels[n] == "CALL":
                        if m := re.match(r"<operator>\.(.*)", names[n]):
                            operator_name = m.group(1)
                            if operator_name not in ("indirection",):
                                fields.append(("operator", n, operator_name))
                        # handle API call
                        else:
                            fields.append(("api", n, names[n]))
            except Exception:
                print("node error", node_id, traceback.format_exc())
                if raise_all:
                    raise
            return fields

        # nx.set_node_attributes(
        #     ast,
        #     {n: f"{n}: {attr['code']}" for n, attr in ast.nodes(data=True)},
        #     "label",
        # )
        # A = nx.drawing.nx_agraph.to_agraph(ast)
        # A.layout("dot")
        # A.draw("abcd.png")

        # n = n.rename(columns={"id": "node_id"})
        n["graph_id"] = graph_id
        # print("select nodes")
        # print(n["node_id"].isin(n for n, attr in cpg.nodes(data=True) if is_decl(attr)))
        decls = n[
            n["node_id"].isin(n for n, attr in cpg.nodes(data=True) if is_decl(attr))
        ].copy()
        decls["fields"] = decls["node_id"].apply(grab_declfeats)
        decls = decls.explode("fields").dropna()
        if verbose: print("extracted fields:", decls["fields"], sep="\n")
        if len(decls) > 0:
            decls["subkey"], decls["subkey_node_id"], decls["subkey_text"] = zip(
                *decls["fields"]
            )
        else:
            decls["subkey"] = None
            decls["subkey_node_id"] = None
            decls["subkey_text"] = None
        return decls
    except Exception:
        print("graph error", graph_id, traceback.format_exc())
        if raise_all:
            raise



