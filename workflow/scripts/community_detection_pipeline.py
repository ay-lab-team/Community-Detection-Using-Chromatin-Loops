import os
import pandas as pd
import argparse
from modules import annotate
from modules import run_multi_community_detection
# import pybedtools as pbt
# pbt.set_bedtools_path('/mnt/BioApps/bedtools/bin/')

# Create a commandline interface #
parser = argparse.ArgumentParser(description="Community detection")

parser.add_argument("-s", "--sample-name", required=True, type=str)
parser.add_argument("-c", "--chipseq", required=True, type=str)
parser.add_argument("-t", "--tss", required=True, type=str)
parser.add_argument("-l", "--loops", required=True, type=str)
# parser.add_argument("-g", "--genome", required=True, type=str)
parser.add_argument(
    "-a",
    "--algorithm",
    type=str,
    required=False,
    choices=["leiden", "louvain"],
    default="leiden",
)
parser.add_argument(
    "-ch",
    "--chromosome",
    type=str,
    required=True,
    choices=list(map(str, list(range(1, 23)))) + ["X"]
)
parser.add_argument("-o", "--outfolder", type=str, required=True)
parser.add_argument("--crank", default="crank", required=False, type=str)
args = parser.parse_args()

# make the output directory
os.makedirs(args.outfolder, exist_ok=True)

# read in the loops file
print("args.loops:", args.loops)
loops_df = pd.read_csv(args.loops, sep='\t')

# read in the TSS file
tss_df = pd.read_csv(args.tss, sep='\t', header=None, names=['chr', 'start', 'end', 'gene_id', 'gene_name'])

# read in the chipseq file
chipseq_df = pd.read_csv(args.chipseq, sep='\t', header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand', 'signalValue', 'qValue', 'pValue', 'peak'])

# annotate the loops
df_processed = annotate(loops_df, tss_df, chipseq_df)

fn = os.path.join(args.outfolder, 'network.annotated.txt')
df_processed.to_csv(fn, sep='\t', index=False)

# run the rest of the pipeline
run_multi_community_detection(
    sample_name=args.sample_name,
    df_processed=df_processed,
    workdir=args.outfolder,
    chrom=args.chromosome,
    algorithm=args.algorithm,
    crank_path=args.crank  # "/home/calimarandi/crank/crank/crank",
)