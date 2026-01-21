library(clusterProfiler)
library(org.Hs.eg.db)

for (class in 1:3){

###### READ GENE ######
gene <- read.table(paste0('Class_',class,'_genelist.txt'), header=T)
gene <- gene$Smybol

###### GET ENTREZID ######
gene.df <- bitr(gene, fromType="SYMBOL",toType="ENTREZID", OrgDb = "org.Hs.eg.db")

###### GO ######
enrich.go <- enrichGO(gene = gene.df$ENTREZID,
                      OrgDb = 'org.Hs.eg.db',
                      ont = 'ALL',
                      pAdjustMethod = 'fdr',
                      pvalueCutoff = 1,
                      qvalueCutoff = 1,
                      readable = T)
enrich.go <- data.frame(enrich.go)
out <- paste0('Class_',class,'_GO_enrich.csv')
write.csv(enrich.go,out,row.names = F)

}
