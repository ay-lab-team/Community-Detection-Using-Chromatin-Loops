from utils import Algorithm
from networkx.classes.graph import Graph
from cdlib import algorithms
from cdlib.classes.node_clustering import NodeClustering
from pyvis.network import Network
import pandas as pd
import numpy as np
import networkx as nx
import subprocess
import os
# import pybedtools as pbt
from IPython.display import display, HTML
# pbt.set_bedtools_path('/mnt/BioApps/bedtools/bin/')

COLORS = [
    '#e6194B',  # red
    '#3cb44b',  # green
    '#ffe119',  # yellow
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#42d4f4',  # cyan
    '#f032e6',  # magenta
    '#bfef45',  # lime
    '#fabed4',  # pink
    '#469990',  # teal
    '#dcbeff',  # lavender
    '#9A6324',  # brown
    '#fffac8',  # beige
    '#800000',  # maroon
    '#aaffc3',  # mint
    '#808000',  # olive
    '#ffd8b1',  # apricot
    '#000075',  # navy
    '#a9a9a9',  # grey
    '#ffffff',  # white
    '#000000',  # black
    '#008080',  # teal
    '#7fffd4',  # aquamarine
    '#ff7f50',  # coral
    '#6495ed',  # cornflower blue
    '#b8860b',  # dark goldenrod
    '#ee82ee',  # violet
    '#da70d6',  # orchid
    '#ff1493',  # deep pink
    '#7fff00',  # chartreuse
    '#ffd700',  # gold
    '#00ffff',  # aqua
    '#4b0082',  # indigo
    '#f5deb3',  # wheat
    '#556b2f',  # dark olive green
    '#d2691e',  # chocolate
    '#ff4500',  # orange-red
    '#8a2be2',  # blue-violet
    '#b22222',  # fire brick
    '#fffff0',  # ivory
    '#f0e68c',  # khaki
    '#90ee90',  # light green
    '#d3d3d3',  # light gray
    '#c71585',  # medium violet red
    '#1e90ff',  # dodger blue
    '#cd5c5c',  # indian red
    '#7b68ee',  # medium slate blue
    '#bdb76b',  # dark khaki
    '#9400d3',  # dark violet
    '#ff6347',  # tomato
    '#008000',  # green
    '#ff00ff',  # fuchsia
    '#00ff7f',  # spring green
    '#dda0dd',  # plum
    '#b0c4de',  # light steel blue
    '#ff69b4',  # hot pink
    '#a0522d',  # sienna
    '#7cfc00',  # lawn green
    '#f08080',  # light coral
    '#87cefa',  # light sky blue
    '#ffefd5',  # papaya whip
    '#00fa9a',  # medium spring green
    '#d8bfd8',  # thistle
    '#6a5acd',  # slate blue
    '#f4a460',  # sandy brown
    '#afeeee',  # pale turquoise
    '#cd853f',  # peru
    '#00ced1',  # dark turquoise
    '#48d1cc',  # medium turquoise
    '#2e8b57',  # sea green
    '#f5f5dc'   # beige
]

SHAPES = {
    "E": "dot",
    "P": "square",
    "EP": "square",  # star
    "PE": "square",  # star
    "O": "triangle",
}


def detect(
    graph: Graph,
    weight: str,
    algorithm: str = Algorithm.leiden.name,
) -> NodeClustering:

    if algorithm == Algorithm.leiden.name:
        community = algorithms.leiden(graph, weights=weight)
    elif algorithm == Algorithm.louvain.name:
        community = algorithms.louvain(graph, weight=weight)
    else:
        raise Exception(f"Not a valid algorithm: {algorithm}")

    return community


def check_overlap(
    anchor_start: int,
    anchor_end: int,
    peak_start: int,
    peak_end: int,
    overlap_tolerance: int,
) -> bool:
    anchor_start_adjusted = anchor_start - overlap_tolerance
    anchor_end_adjusted = anchor_end + overlap_tolerance
    overlap = (
        (anchor_start_adjusted <= peak_start <= anchor_end_adjusted)
        or (anchor_start_adjusted <= peak_end <= anchor_end_adjusted)
    )
    return overlap


def annotate(
    loops_df: pd.DataFrame,
    tss_df: pd.DataFrame,
    chipseq_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Annotate the loop anchors as P, E, PE, or O and returns a Pandas Dataframe for further processing.

    Params
    ------
    loops: Pandas dataframe which has the loops information
    tss_df: Pandas dataframe which cointains the tss information 
    chipseq_df: Pandas dataframe which cointains the chipseq information

    Return
    ------
    Pandas dataframe that contains all annotations for every loop.
    [source, target, source_annotation, target_annotation, combined_annotation, -log10-Q-Value_Bias, q_norm]
    source: A region on the chromosome (Anchor1)
    target: Another region on the chromosome (Anchor2)
    source_annotation: The annotation for Anchor1 (Either P, E, or PE)
    target_annotation: The annotation for Anchor2 (Either P, E, or PE)
    combined_annotation: Pair annotation for source_annotation and target_annotation (such as E-E, P-P)
    -log10-Q-Value_Bias: -log10-Q-value value conversion of the q-value for each loop
    q_norm: Q-value for each loop fro FitHiChip
    """

    # tss
    def annotate_bin_tss(loops_chrom, loops_start, loops_end, tss_dict):
        tss_chrom = tss_dict.get(loops_chrom, None)
        if tss_chrom is None:
            return 'O', f'{loops_chrom}-{loops_start}-{loops_end}' 
        tss_start = tss_chrom['start']
        tss_end = tss_chrom['end']
        tss_gene_name = tss_chrom['gene_name']
     
        mask1 = np.logical_and((tss_start - 2500) <= loops_start, loops_start <= (tss_end + 2500))
        mask2 = np.logical_and((tss_start - 2500) <= loops_end, loops_end <= (tss_end + 2500))
        mask3 = np.logical_and(loops_end > (tss_end + 2500), (tss_start - 2500) > loops_start)
        mask = np.logical_or(np.logical_or(mask1, mask2), mask3)
        
        if np.any(mask):
            gene_names = ','.join(tss_gene_name[mask])
            return 'P', gene_names
        else:
            return 'O', f'{loops_chrom}-{loops_start}-{loops_end}'

    tss_dict = tss_df.groupby('chr').apply(lambda x: {'start': x['start'].values,
                                                    'end': x['end'].values,
                                                    'gene_name': x['gene_name'].values}).to_dict()

    output_loops_tss = loops_df.copy()
    print(loops_df)
    print(len(loops_df))
    #print(output_loops_tss)  # TEST
    results = [annotate_bin_tss(row['chr1'], row['s1'], row['e1'], tss_dict) for _, row in output_loops_tss.iterrows()]
    print(len(results))
    print(len(output_loops_tss))
    print(output_loops_tss)


    output_loops_tss[['left_anchor_annotation', 'left_anchor_gene_name']] = np.array([annotate_bin_tss(row['chr1'], row['s1'], row['e1'], tss_dict) for _, row in output_loops_tss.iterrows()])
    output_loops_tss[['right_anchor_annotation', 'right_anchor_gene_name']] = np.array([annotate_bin_tss(row['chr2'], row['s2'], row['e2'], tss_dict) for _, row in output_loops_tss.iterrows()])
    output_loops_tss = output_loops_tss.reindex(columns=['chr1', 's1', 'e1', 'chr2', 's2', 'e2', 'left_anchor_annotation', 'right_anchor_annotation', 'left_anchor_gene_name', 'right_anchor_gene_name'])

    output_loops_tss["Q-Value_Bias"] = loops_df["Q-Value_Bias"]

    output_loops_tss["-log10-Q-Value_Bias"] = - np.log10(output_loops_tss["Q-Value_Bias"])

    # normalizing the q-value from fithichip and adding the column for it
    Q = output_loops_tss["-log10-Q-Value_Bias"]
    Q_clipped = np.minimum(Q, 20)
    q_norm = (np.array(Q_clipped) - np.min(Q_clipped)) / (np.max(Q_clipped) - np.min(Q_clipped))
    output_loops_tss["q_norm"] = q_norm

    loops_tss_anno_agg = output_loops_tss.copy()

    # Chipseq

    def annotate_bin_chipseq(loops_chrom, loops_start, loops_end, chipseq_dict):
        chipseq_chrom = chipseq_dict.get(loops_chrom, None)
        if chipseq_chrom is None:
            return 'O', f'{loops_chrom}-{loops_start}-{loops_end}'
        
        chipseq_start = chipseq_chrom['start']
        chipseq_end = chipseq_chrom['end']
        
        mask1 = np.logical_and(chipseq_start <= loops_start, loops_start <= chipseq_end)
        mask2 = np.logical_and(chipseq_start <= loops_end, loops_end <= chipseq_end)
        mask3 = np.logical_and(loops_end > chipseq_end, chipseq_start > loops_start)
        mask = np.logical_or(np.logical_or(mask1, mask2), mask3)
        
        if np.any(mask):
            return 'E', f'{loops_chrom}-{loops_start}-{loops_end}'
        else:
            return 'O', f'{loops_chrom}-{loops_start}-{loops_end}'
        
    chipseq_dict = chipseq_df.groupby('chrom').apply(lambda x: {'start': x['start'].values,
                                                    'end': x['end'].values}).to_dict()

    loops_chipseq_anno = loops_df.copy()
    loops_chipseq_anno[['left_anchor_annotation', 'left_anchor_gene_name']] = np.array([annotate_bin_chipseq(row['chr1'], row['s1'], row['e1'], chipseq_dict) for _, row in loops_chipseq_anno.iterrows()])
    loops_chipseq_anno[['right_anchor_annotation', 'right_anchor_gene_name']] = np.array([annotate_bin_chipseq(row['chr2'], row['s2'], row['e2'], chipseq_dict) for _, row in loops_chipseq_anno.iterrows()])
    loops_chipseq_anno = loops_chipseq_anno.reindex(columns=['chr1', 's1', 'e1', 'chr2', 's2', 'e2', 'left_anchor_annotation', 'right_anchor_annotation', 'left_anchor_gene_name', 'right_anchor_gene_name'])

    loops_chipseq_anno_agg = loops_chipseq_anno.copy()

    loops_chipseq_anno_agg["Q-Value_Bias"] = loops_df["Q-Value_Bias"]

    loops_chipseq_anno_agg["-log10-Q-Value_Bias"] = - np.log10(loops_chipseq_anno_agg["Q-Value_Bias"])

    # normalizing the q-value from fithichip and adding the column for it
    Q = loops_chipseq_anno_agg["-log10-Q-Value_Bias"]
    Q_clipped = np.minimum(Q, 20)
    q_norm = (np.array(Q_clipped) - np.min(Q_clipped)) / (np.max(Q_clipped) - np.min(Q_clipped))
    loops_chipseq_anno_agg["q_norm"] = q_norm

    loops_chipseq_anno_agg.reset_index(drop=True, inplace=True)

    df_processed = loops_tss_anno_agg.iloc[:, :6].copy()

    df_processed["left_anchor_annotation_tss"] = loops_tss_anno_agg["left_anchor_annotation"]
    df_processed["right_anchor_annotation_tss"] = loops_tss_anno_agg["right_anchor_annotation"]

    df_processed["left_anchor_gene_name_tss"] = loops_tss_anno_agg["left_anchor_gene_name"]
    df_processed["right_anchor_gene_name_tss"] = loops_tss_anno_agg["right_anchor_gene_name"]

    df_processed["left_anchor_annotation_chipseq"] = loops_chipseq_anno_agg["left_anchor_annotation"]
    df_processed["right_anchor_annotation_chipseq"] = loops_chipseq_anno_agg["right_anchor_annotation"]

    df_processed["left_anchor_gene_name_chipseq"] = loops_chipseq_anno_agg["left_anchor_gene_name"]
    df_processed["right_anchor_gene_name_chipseq"] = loops_chipseq_anno_agg["right_anchor_gene_name"]

    #df_processed.to_csv('df_processed_new.csv', sep='\t', index=False)

    def get_annotation(row):
        left_tss = row['left_anchor_annotation_tss']
        left_chip = row['left_anchor_annotation_chipseq']
        right_tss = row['right_anchor_annotation_tss']
        right_chip = row['right_anchor_annotation_chipseq']
        
        if 'P' in left_tss:
            left_anno = 'P'
        elif 'E' in left_chip:
            left_anno = 'E'
        else:
            left_anno = 'O'
        
        if 'P' in right_tss:
            right_anno = 'P'
        elif 'E' in right_chip:
            right_anno = 'E'
        else:
            right_anno = 'O'
        
        return pd.Series({'left_anchor_annotation': left_anno, 'right_anchor_annotation': right_anno})

    df_processed[['source_annotation', 'target_annotation']] = df_processed.apply(get_annotation, axis=1)

    def get_gene_name(row):
        left_anno = row['source_annotation']
        left_tss = row['left_anchor_gene_name_tss']
        left_chip = row['left_anchor_gene_name_chipseq']
        right_anno = row['target_annotation']
        right_tss = row['right_anchor_gene_name_tss']
        right_chip = row['right_anchor_gene_name_chipseq']
        
        if 'P' in left_anno:
            left_gene = left_tss
        else:
            left_gene = left_chip
        
        if 'P' in right_anno:
            right_gene = right_tss
        else:
            right_gene = right_chip
        
        return pd.Series({'left_anchor_gene_name': left_gene, 'right_anchor_gene_name': right_gene})

    df_processed[['source_gene_name', 'target_gene_name']] = df_processed.apply(get_gene_name, axis=1)

    df_processed = df_processed.drop(['left_anchor_annotation_tss', 'right_anchor_annotation_tss', 'left_anchor_gene_name_tss', 'right_anchor_gene_name_tss', 'left_anchor_annotation_chipseq', 'right_anchor_annotation_chipseq', 'left_anchor_gene_name_chipseq', 'right_anchor_gene_name_chipseq'], axis=1)

    df_processed["Q-Value_Bias"] = loops_chipseq_anno_agg["Q-Value_Bias"]
    df_processed["-log10-Q-Value_Bias"] = loops_chipseq_anno_agg["-log10-Q-Value_Bias"]
    df_processed["q_norm"] = loops_chipseq_anno_agg["q_norm"]

    df_processed[['source', 'target']] = df_processed.apply(lambda x: pd.Series([f"{x['chr1']}-{x['s1']}-{x['e1']}", f"{x['chr2']}-{x['s2']}-{x['e2']}"]), axis=1)
    df_processed = df_processed.drop(columns=['chr1', 's1', 'e1', 'chr2', 's2', 'e2'])

    df_processed = df_processed.reindex(columns=['source','target','source_annotation', 'target_annotation', 'source_gene_name', 'target_gene_name', 'Q-Value_Bias', '-log10-Q-Value_Bias', 'q_norm'])

    return df_processed

def crank(
    df_filtered: pd.DataFrame,
    community: NodeClustering,
    network_file: str,
    community_file: str,
    crank_scores_file: str,
    crank_summary_file: str,
    crank_path: str = 'crank',
    comm_type: str = 'Community'
):
    """
    CRank works by taking a pandas dataframe as an input and converts the necessary information into
    two main files which become inputs of the CRank tool. Then the CRank tool is run on these files creating an
    output file which contains the crank scores. No variables are returned but output files are generated under
    {workdir}/{file_name}.

    Params
    ------
    df_filtered: pd.DataFrame
    community: NodeClustering
    network_file: str
        Filename for the network file
    community_file: str
        Filename for the community file
    crank_scores_file: str
        Filename for crank scores
    crank_summary_file: str
        Filename for crank std output, summarizes the crank process
    crank_path:
        Path to the CRank program; combined with workdir
    workdir:
        Output directory
    """

    # define the comm prefix
    if comm_type == 'Community':
        comm_prefix = 'Cmt'
    elif comm_type == 'Subcommunity':
        comm_prefix = 'SubCmt'
    else:
        raise Exception("comm_type: {} is not an element of ['Community', 'Subcommunity'].".format(comm_type))

    # nodes file for crank
    df_filtered[["source", "target"]].to_csv(
        network_file,
        sep="\t",
        header=False,
        index=False
    )

    # communities file for crank
    file = open(community_file, "w")
    length = len(community)
    for ind in range(length):
        file.write(f"{comm_prefix}{ind + 1} {' '.join(community[ind])}")
        if ind != (length - 1):
            file.write("\n")
    file.close()

    # run crank from the command line using the previously created files which crank requires
    # std output is also written since some summary information is output
    crank_cmd = [f"{crank_path}", f"-i:{network_file}", f"-c:{community_file}", f"-o:{crank_scores_file}"]
    stdout = subprocess.check_output(crank_cmd)
    with open(crank_summary_file, 'w') as fw:
        fw.write(stdout.decode())


def analyze_communities(
    df_processed: pd.DataFrame,    # need to modify based per chromosome and the chosen algorithm
    chrom: str,
    algorithm: str = Algorithm.leiden.name,
    crank_path: str = 'crank',
    workdir: str = './'
):

    """
    Generate a graph from the given loops data as well as the specific chromosome.
    Then, run a community detection algorithm and determine the 1st layer communities.
    Finally, return a dataframe with the communities (which is an input to another function).

    Params
    ------
    df_processed: pd.DataFrame
        A dataframe where each entry represents nodes, their edges and additional annotation information
        (columns equals: "source", "target", "source_annotation", "target_annotation", "combined_annotation",
        "-log10-Q-Value_Bias")
    chrom: str
    algorithm: str
    crank_path: str
    workdir: str
    """

    # filter for selected chromosome
    df_filtered = df_processed[
        df_processed["source"].str.startswith("chr" + str(chrom) + "-") &
        df_processed["target"].str.startswith("chr" + str(chrom) + "-")
    ]

    # generate a network from the df
    G = nx.from_pandas_edgelist(
        df=df_filtered,
        source="source",
        target="target",
        edge_attr=["q_norm"],
    )

    # detect communities
    communities = detect(graph=G, weight="q_norm", algorithm=algorithm).communities

    # count the number of nodes for each community
    comm_sizes = []
    for comm in communities:
        comm_sizes.append(len(comm))

    # calculate the crank of the communities & parse
    network_file = os.path.join(workdir, "network.txt")
    community_file = os.path.join(workdir, "community.txt")
    crank_scores_file = os.path.join(workdir, "crank_scores.txt")
    crank_summary_file = os.path.join(workdir, 'crank.summary.txt')
    crank(
        df_filtered=df_filtered,
        community=communities,
        network_file=network_file,
        community_file=community_file,
        crank_scores_file=crank_scores_file,
        crank_summary_file=crank_summary_file,
        crank_path=crank_path,
        comm_type='Community'
    )
    crank_fn = os.path.join(workdir, "crank_scores.txt")
    df_crank_score = pd.read_csv(crank_fn, sep="\t")
    df_crank_score["node_count"] = comm_sizes

    return df_crank_score, df_filtered, communities, network_file, community_file, crank_scores_file, crank_summary_file, comm_sizes

def create_html_link(sr, network_file_name, subcommunity_file_name):
    # Check if 'Community' attribute exists in sr object
    if 'Community' not in sr:
        return '', '', ''

    # rename & get path
    rename_comm = sr['Community'].replace('Cmt', 'comm')
    community_path = '{}/{}.html'.format(rename_comm, rename_comm)
    network_path = '{}/{}'.format(rename_comm, network_file_name)
    subcommunity_path = '{}/{}'.format(rename_comm, subcommunity_file_name)

    # create the community visualization link
    community_link_tmpl = '<a href="{}">community_visualization<a>'
    community_link = community_link_tmpl.format(community_path)

    # create the network file link
    network_link_tmpl = '<a href="{}">network_file<a>'
    network_link = network_link_tmpl.format(network_path)

    # create subcommunity file link
    subcommunity_link_tmpl = '<a href="{}">subcommunity_file<a>'
    subcommunity_link = subcommunity_link_tmpl.format(subcommunity_path)

    return community_link, network_link, subcommunity_link


def extract_crank_and_network_info(sample_name, chrom, network_fn, community_fn, crank_scores_file, comm_type, comm_sizes):
    """
    Extract crank and network information. Allows specifying the community or subcommunity levels.

    Params:
    ------
    comm_type: str
        Choose between 'community' or 'subcommunity' to create the relevant table.

    Returns:
    -------
    """

    # Check the comm_type for correctness
    if comm_type not in ['Community', 'Subcommunity']:
        raise Exception("comm_type: {} is not an element of ['Community', 'Subcommunity'].".format(comm_type))

    # Extract network information

    # Read in the network file
    with open(network_fn, 'r') as f:
        edges = [line.strip().split('\t') for line in f.readlines()]

    # Construct the graph
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    # Read in the community file
    with open(community_fn, 'r') as f:
        communities = [line.strip().split() for line in f.readlines()]

    # Iterate through the communities and count nodes and edges in each one
    results = []
    subcommunity_edges = []
    for community in communities:

        # Get nodes
        nodes = set(community[1:])

        # Get subgraph of community
        subgraph = G.subgraph(list(nodes))

        # Count edges
        num_edges = len(subgraph.edges)

        # Keep track of edges within communities
        subcommunity_edges.extend(list(subgraph.edges))

        # Save metadata
        name = community[0]
        results.append((name, len(nodes), num_edges))

    # Convert the results to a pandas DataFrame
    comm_df = pd.DataFrame(results, columns=[comm_type, 'Nodes', 'Edges'])

    # Compare edges across the entire network versus those within the community
    network_edges = list(G.edges)
    between_comm_edges = set(network_edges).difference(subcommunity_edges)

    # Load crank scores (community + scores)
    crank_df = pd.read_table(crank_scores_file)

    if comm_sizes:
        crank_df["node_count"] = comm_sizes
        crank_df = crank_df[crank_df["node_count"] > 10]
        crank_df.drop("node_count", axis=1, inplace=True)

    crank_df.rename(columns={'Community': comm_type}, inplace=True)

    # Merge the data
    merged_df = crank_df.merge(comm_df, on=comm_type)

    # Add chrom column
    merged_df['Chrom'] = chrom

    # Create the HTML links
    merged_df['Community_Visualization'], merged_df['Network_File'], merged_df['Subcommunity_File'] = zip(*merged_df.apply(lambda row: create_html_link(row, 'network.txt', 'community.txt'), axis=1))

    return merged_df, between_comm_edges



def analyze_subcommunities(sample_name, chrom, comm_no, df_filtered, communities, algorithm, crank_path, workdir='./'):
    """
    Similar to analyze_communities, generate a graph from each community and apply community detection once again
    in order to get the subcommunities (second layer of community detection).

    Params
    ------
    comm_no: pd.DataFrame
    df_filtered: pd.DataFrame
        df_processed: pd.DataFrame
        A dataframe where each entry represents nodes, their edges and additional annotation information
        (columns equals: "source", "target", "source_annotation", "target_annotation", "combined_annotation",
        "-log10-Q-Value_Bias"). df_filtered is the chromosome filtered version of the df_processed dataframe.
    """

    comm_current = communities[comm_no]

    # filter the data set to extract the current community
    df_comm_current = df_filtered[
        df_filtered["source"].isin(comm_current) &
        df_filtered["target"].isin(comm_current)
    ]

    # create the subnetwork of the community
    G_comm_current = nx.from_pandas_edgelist(
        df=df_comm_current,
        source="source",
        target="target",
        edge_attr=["q_norm"],
    )

    # catching detection failures due to no subcommunities
    try:
        subcommunities = detect(
            graph=G_comm_current,
            weight="q_norm",
            algorithm=algorithm
        ).communities
    except Exception as e:
        print("Error detected: {}".format(e))

    ########### Run CRank #########
    # calculate the crank of the communities
    network_file = os.path.join(workdir, "network.txt")
    community_file = os.path.join(workdir, 'community.txt')
    crank_scores_file = os.path.join(workdir, "crank_scores.txt")
    crank_summary_file = os.path.join(workdir, "crank.summary.txt")

    crank(
        df_filtered=df_comm_current,
        community=subcommunities,
        network_file=network_file,
        community_file=community_file,
        crank_scores_file=crank_scores_file,
        crank_summary_file=crank_summary_file,
        crank_path=crank_path,
        comm_type='Subcommunity'
    )

    return df_comm_current, subcommunities, network_file, community_file, crank_scores_file, crank_summary_file


def visualize_subcommunities(sample_name, chrom, df_comm_current, subcommunities, comm_no, workdir):
    """

    Params
    ------
    df_comm_current: pd.DataFrame
    subcommunities:

    Returns
    -------
    """
    # create a dataframe for network viz, shapes and colors added as well
    col_names = ["anchor", "anchor_gene_name", "shape"]
    df = pd.concat(
        [
            df_comm_current.iloc[:, [0, 4, 2]].set_axis(col_names, axis="columns"),
            df_comm_current.iloc[:, [1, 5, 3]].set_axis(col_names, axis="columns")
        ]
    ).drop_duplicates()

    # df.to_csv('df_pre.csv', sep='\t', index=False) # print and continue 

    def modify_values(value):
        if value.startswith('chr'):
            parts = value.split("-")
            num1 = str(int(parts[1]) // 1000) + "kb"
            num2 = str(int(parts[2]) // 1000) + "kb"
            return parts[0] + "-" + num1 + "-" + num2
        else:
            return value

    # apply the function to the 'anchor_gene_name' column
    df['anchor_gene_name'] = df['anchor_gene_name'].apply(modify_values)
    # df.to_csv('df_post.csv', sep='\t', index=False) # print and continue 

    # convert the original shapes to the final shapes column
    df = df.replace({"shape": SHAPES}).reset_index()

    # add dcolor to each of the anchors
    for idx, comm in enumerate(subcommunities):
        df.loc[df["anchor"].isin(comm), "color"] = COLORS[idx]

    # convert to a network
    net = Network()
    net.force_atlas_2based()
    net.set_options = {"""
    "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "opacity": 1,
        "font": {
        "size": 16
        },
        "size": null
    },
    "edges": {
        "color": {
        "inherit": true
        },
        "selfReferenceSize": null,
        "selfReference": {
        "angle": 0.7853981633974483
        },
        "smooth": {
        "type": "cubicBezier"
        }
    },
    "physics": {
        "repulsion": {
        "centralGravity": 0.4
        },
        "minVelocity": 0.75,
        "solver": "repulsion",
        "timestep": 0.56
    }
    }
    """}

    net.add_nodes(df["anchor"], label=df["anchor_gene_name"], title=df["anchor_gene_name"], color=df["color"], shape=df["shape"])
    # net.add_nodes(df["anchor"], title=df["anchor"], color=df["color"], shape=df["shape"])
    for idx, row in df_comm_current.iterrows():
        net.add_edge(row["source"], row["target"], value=row["q_norm"], title=row["q_norm"], physics=False)

    # annotate neighbooring nodes
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        neighbor_names = []
        for neighbor_id in neighbor_map[node["id"]]:
            neighbor_names.append(df.loc[df["anchor"] == neighbor_id, "anchor_gene_name"].iloc[0])
        node["title"] += "\n\nNeighbors:\n" + "\n".join(neighbor_names)
        node["value"] = len(neighbor_map[node["id"]])


    # extract network information
    network_file = os.path.join(workdir, "network.txt")
    community_file = os.path.join(workdir, 'community.txt')
    crank_scores_file = os.path.join(workdir, "crank_scores.txt")

    comm_network_props, comm_between_edges = extract_crank_and_network_info(
        sample_name,
        chrom,
        network_file,
        community_file,
        crank_scores_file,
        'Subcommunity',
        None
        )
    html_fn = os.path.join(workdir, f"comm{comm_no + 1}.html")

    sub_template = open('workflow/scripts/subcommunity.template.html').read()

    comm_network_props = comm_network_props.drop('Subcommunity_File', axis=1)

    df = comm_network_props.drop(["Chrom", "Community_Visualization", "Network_File"], axis=1)

    # Assuming df is your DataFrame
    df["SubCommunity_File"] = '<a href="community.txt">' + "Community_File" + '</a>'
    
    COPY_COLORS = list(COLORS)
    
    # Slice the list to match the length of the DataFrame
    COPY_COLORS = COPY_COLORS[:len(df)]

    # Add the new column as the first column in the DataFrame
    df.insert(0, 'Color', COPY_COLORS)

    data_html = df.to_html(render_links=True, escape=False, table_id="CD_Table")
    with open(html_fn, 'w') as fw:
        html_str = sub_template.format(sample=sample_name,
                                    chrom=chrom,
                                    network=net.generate_html(html_fn),
                                    datatable=data_html,
                                    num_edges_between=len(comm_between_edges))
        # print(html_str)  # Add this line to check the generated HTML string
        fw.write(html_str)





def run_multi_community_detection(sample_name, df_processed, workdir, chrom, algorithm, crank_path):
    """
    Params
    ------
    """

    # 1) return files in analyze
    # 2) within pipeline, pass samplename

    ########## Community Level ##########
    # analyze communities
    df_crank_score, df_filtered, communities, network_file, community_file, crank_scores_file, crank_summary_file, comm_sizes = \
        analyze_communities(df_processed, chrom, algorithm, crank_path, workdir)

    # create community level html
    summary_html = os.path.join(workdir, 'summary.html')
    chrom_str = 'chr' + chrom  # chr added to chrom
    comm_network_props, comm_between_edges = extract_crank_and_network_info(sample_name, chrom_str, network_file, community_file, crank_scores_file, 'Community', comm_sizes)
    community_html_template = open('workflow/scripts/community.template.html').read()

    comm_network_props = comm_network_props.drop('Subcommunity_File', axis=1)

    # Add a dropdown arrow for each column
    datatable = comm_network_props.to_html(render_links=True, escape=False, table_id="CD_Table1")

    final_html_str = community_html_template.format(sample=sample_name,
                                                    chrom=chrom,
                                                    datatable=datatable,
                                                    num_edges_between=len(comm_between_edges))

    with open(summary_html, 'w') as fw:
        fw.write(final_html_str)


    ########## Subcommunity level ##########

    # creates a subcommunity html

    # cycle through the communities
    length = df_crank_score.shape[0]
    for comm_no in range(length):

        # create the output file + workdir
        subdir = os.path.join(workdir, 'comm{}/'.format(comm_no + 1))
        os.makedirs(subdir, exist_ok=True)

        # analyze
        df_comm_current, subcommunities, network_file, community_file, \
            crank_scores_file, crank_summary_file = \
            analyze_subcommunities(sample_name, chrom_str, comm_no, df_filtered, communities, algorithm, crank_path, subdir)

        # visualize the current community with subcommunities
        visualize_subcommunities(sample_name, chrom, df_comm_current, subcommunities, comm_no, subdir)
        # break