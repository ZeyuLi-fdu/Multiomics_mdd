import pandas as pd
import numpy as np
from IPython.display import SVG, display
from sknetwork.clustering import Louvain, Leiden, KCenters, PropagationClustering,  get_modularity
from sknetwork.data import from_edge_list
from sknetwork.ranking import PageRank
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction

network_datdir="./RESULTS/Omics_network/"
target = 'MDD'

# inter-omics only =====

mydf = pd.read_csv(network_datdir+target+'_SigAss_Interomics.csv')

print(mydf.shape)
edge_list, weighted_edge_list = [], []

for i in range(len(mydf)):
    edge_list.append((mydf.var1.iloc[i], mydf.var2.iloc[i]))
    weighted_edge_list.append((mydf.var1.iloc[i], mydf.var2.iloc[i], abs(mydf.t.iloc[i])))

graph = from_edge_list(weighted_edge_list)
adjacency = graph.adjacency
names = graph.names
louvain = Louvain(shuffle_nodes = False)
louvain_fit = louvain.fit_predict(adjacency)
labels, counts = np.unique(louvain_fit, return_counts=True)
print(labels, counts)
mod_Louvain = get_modularity(adjacency, louvain_fit)
print("modularity Louvain: " + str(mod_Louvain))
lou_cls_df = pd.DataFrame({'OmicFeature':names, 'louvain_cls':louvain_fit})

pagerank_all = PageRank()
scores_all = pagerank_all.fit_predict(adjacency)
rank_all_df = pd.DataFrame({'OmicFeature':names, 'PageRank_Score':scores_all})


lou_cls_lst = list(set(lou_cls_df.louvain_cls))
rank_df = pd.DataFrame()

for lou_cls in lou_cls_lst:
    cls_f_lst = lou_cls_df.loc[lou_cls_df.louvain_cls == lou_cls].OmicFeature.tolist()
    select_idx = []
    for i in range(len(mydf)):
        var1, var2 = mydf.var1.iloc[i], mydf.var2.iloc[i]
        if ((var1 in cls_f_lst) & (var2 in cls_f_lst)):
            select_idx.append(i)
        else:
            pass
    cls_ass_df = mydf.iloc[select_idx, :]
    cls_ass_df.reset_index(inplace = True, drop = True)
    print(len(cls_ass_df))
    weighted_edge_list = []
    for i in range(len(cls_ass_df)):
        weighted_edge_list.append((cls_ass_df.var1.iloc[i], cls_ass_df.var2.iloc[i], abs(cls_ass_df.t.iloc[i])))
    graph = from_edge_list(weighted_edge_list)
    adjacency = graph.adjacency
    names = graph.names
    pagerank = PageRank()
    scores = pagerank.fit_predict(adjacency)
    cls_rank_df = pd.DataFrame({'OmicFeature': names, 'PageRank_Score_cls'+str(lou_cls): scores})
    rank_df = pd.concat([rank_df, cls_rank_df], axis = 0)

gdf = pd.merge(lou_cls_df, rank_all_df, how = 'inner', on = 'OmicFeature')
gdf = pd.merge(gdf, rank_df, how = 'inner', on = 'OmicFeature')

gdf.to_csv(network_datdir+'/NetworkAnalysis_1009/Louvain_'+target+'_Interomics.csv')
