### Move to cugraph

### Feature Tensor to DataFrame Utils
def convert_to_column_major(t):
    return t.t().contiguous().t()


### Add ndata utils
def add_ndata_of_single_type(gs, node_ids, t, feat_name, ntype):
    ## send it to cuda
    t = t.to("cuda")
    t = convert_to_column_major(t)
    ar = cupy.from_dlpack(zerocopy_to_dlpack(t))
    del t
    df = cudf.DataFrame(ar)
    df.columns = [f"{feat_name}_{i}" for i in range(len(df.columns))]

    node_ids = dgl.backend.zerocopy_to_dlpack(node_ids)
    node_ser = cudf.from_dlpack(node_ids)
    df["node_id"] = node_ser
    gs.add_node_data(df, "node_id", feat_name, ntype=ntype)
    return gs


def add_ndata(gs, graph):
    for feat_name in graph.ndata.keys():
        if graph.ntypes == 1:
            if torch.is_tensor(graph.ndata[feat_name]):
                ntype = graph.ntypes[0]
                t = graph[ndata]
                gs = add_ndata_of_single_type(gs, t, feat_name, ntype)
        else:
            ### Hetrogeneous case
            for ntype in graph.ntypes:
                t = graph.ndata[feat_name][ntype]
                node_ids = graph.nodes(ntype).to("cuda")
                node_ids = gs.dgl_n_id_to_cugraph_id(node_ids, ntype)
                gs = add_ndata_of_single_type(
                    gs, node_ids, t, feat_name, ntype
                )
    return gs


### Add edata utils
def add_edata_single_type(gs, src_t, dst_t, can_etype):

    src_t = src_t.to("cuda")
    dst_t = dst_t.to("cuda")

    df = cudf.DataFrame(
        {
            "src": cudf.from_dlpack(zerocopy_to_dlpack(src_t)),
            "dst": cudf.from_dlpack(zerocopy_to_dlpack(dst_t)),
        }
    )
    gs.add_edge_data(df, ["src", "dst"], feat_name="_ID", etype=can_etype)
    return gs


def add_edata(gs, graph):
    for can_etype in g.canonical_etypes:
        src_t, dst_t = graph.edges(form="uv", etype=can_etype)
        src_type, etype, dst_type = can_etype
        src_t = gs.dgl_n_id_to_cugraph_id(src_t, src_type)
        dst_t = gs.dgl_n_id_to_cugraph_id(dst_t, dst_type)
        add_edata_single_type(gs, src_t, dst_t, can_etype)
