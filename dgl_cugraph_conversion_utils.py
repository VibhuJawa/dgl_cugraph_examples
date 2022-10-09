# Utils to convert b/w dgl hetrograph to cugraph HetroGraph
# TODO: Add upstream
from typing import Optional

import cudf
import cupy as cp
import dgl
import torch
from dgl.backend import zerocopy_to_dlpack


# Feature Tensor to DataFrame Utils
def convert_to_column_major(t: torch.Tensor):
    return t.t().contiguous().t()


# Add ndata utils
def add_ndata_of_single_type(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    feat_t: torch.Tensor,
    feat_name: str,
    ntype: str,
    node_offset: int,
    idtype=torch.int64,
):
    # send it to cuda
    feat_t = feat_t.to("cuda")
    node_ids = (
        dgl.backend.arange(0, feat_t.shape[0], idtype, ctx="cuda")
        + node_offset
    )
    node_ids = cp.from_dlpack(zerocopy_to_dlpack(node_ids))

    feat_t = convert_to_column_major(feat_t)
    ar = cp.from_dlpack(zerocopy_to_dlpack(feat_t))
    del feat_t
    df = cudf.DataFrame(ar)
    df.columns = [f"{feat_name}_{i}" for i in range(len(df.columns))]
    df["node_id"] = node_ids
    gs.add_node_data(df, "node_id", feat_name, ntype=ntype)
    return gs


def add_ndata(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
    num_nodes_dict: Optional[dict[str, int]] = None,
):
    if len(graph.ntypes) > 1:
        if num_nodes_dict is None:
            raise ValueError(
                "num_nodes_dict must be provided for adding ndata from Heterogeneous Graphs"
            )
        mutiple_ntypes = True
        node_id_offset_d = gs._CuGraphStorage__get_node_id_offset_d(
            num_nodes_dict
        )
    else:
        mutiple_ntypes = False

    for feat_name, feat_d in graph.ndata.items():
        if mutiple_ntypes:
            for ntype, feat_t in feat_d.items():
                node_offset = node_id_offset_d[ntype]
                gs = add_ndata_of_single_type(
                    gs,
                    feat_t,
                    feat_name,
                    ntype,
                    node_offset,
                    idtype=graph.idtype,
                )
        else:
            ntype = graph.ntypes[0]
            feat_t = feat_d
            gs = add_ndata_of_single_type(gs, feat_t, feat_name, ntype, 0)

    return gs


# Add edata utils
def add_edges_of_single_type(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    src_t: torch.Tensor,
    dst_t: torch.Tensor,
    can_etype: tuple([str, str, str]),
    src_offset: int,
    dst_offset: int,
):

    src_t = src_t.to("cuda")
    dst_t = dst_t.to("cuda")

    src_type, edge_type, dst_type = can_etype

    src_t = src_t + src_offset
    dst_t = dst_t + dst_offset

    df = cudf.DataFrame(
        {
            "src": cudf.from_dlpack(zerocopy_to_dlpack(src_t)),
            "dst": cudf.from_dlpack(zerocopy_to_dlpack(dst_t)),
        }
    )
    gs.add_edge_data(
        df, ["src", "dst"], feat_name="_ID", canonical_etype=can_etype
    )
    return gs


def add_node_single_type(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    num_nodes: int,
    ntype: str = None,
    node_offset: int = 0,
    idtype=torch.int64,
):
    node_ids = (
        dgl.backend.arange(0, num_nodes, idtype, ctx="cuda") + node_offset
    )
    node_ids = cp.from_dlpack(zerocopy_to_dlpack(node_ids))
    df = cudf.DataFrame({"node_ids": node_ids})
    gs.add_node_data(
        df,
        node_col_name="node_ids",
        ntype=ntype,
        is_single_vector_feature=False,
    )


def add_nodes(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
    num_nodes_dict: Optional[dict[str, int]] = None,
):
    if len(graph.ntypes) > 1:
        if num_nodes_dict is None:
            raise ValueError(
                "num_nodes_dict must be provided for adding edges from Heterogeneous Graphs"
            )
        mutiple_ntypes = True
        node_id_offset_d = gs._CuGraphStorage__get_node_id_offset_d(
            num_nodes_dict
        )
    else:
        mutiple_ntypes = False

    if mutiple_ntypes:
        for ntype in graph.ntypes:
            add_node_single_type(
                gs,
                num_nodes_dict[ntype],
                ntype,
                node_id_offset_d[ntype],
                graph.idtype,
            )
    else:
        add_node_single_type(gs, graph.num_nodes(), ntype, 0, graph.idtype)


def add_edges(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
    num_nodes_dict: Optional[dict[str, int]] = None,
):

    if len(graph.ntypes) > 1:
        if num_nodes_dict is None:
            raise ValueError(
                "num_nodes_dict must be provided for adding edges from Heterogeneous Graphs"
            )
        mutiple_ntypes = True
        node_id_offset_d = gs._CuGraphStorage__get_node_id_offset_d(
            num_nodes_dict
        )
    else:
        mutiple_ntypes = False

    for can_etype in graph.canonical_etypes:
        src_t, dst_t = graph.edges(form="uv", etype=can_etype)
        src_type, etype, dst_type = can_etype
        if mutiple_ntypes:
            src_offset, dst_offset = (
                node_id_offset_d[src_type],
                node_id_offset_d[dst_type],
            )
        else:
            src_offset, dst_offset = 0, 0
        add_edges_of_single_type(
            gs, src_t, dst_t, can_etype, src_offset, dst_offset
        )


# Testing Utils
def assert_same_num_nodes(gs, g):
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == gs.num_nodes(ntype)


def assert_same_num_edges(gs, g):
    for etype in g.etypes:
        assert g.num_edges(etype) == gs.num_edges(etype)
