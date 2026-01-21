library(data.table)

##### Set
args <- commandArgs(trailingOnly = TRUE)
omic_id <- as.numeric(args[1])
year_id <- as.numeric(args[2])

name_omics <- c('Proteome','Clinical','Metabolome')
omic <- name_omics[omic_id]
name_year <- c(-5,-10)
output_year <- c("5_years","10_years")
year <- name_year[year_id]

##### MDD
mdd <- read.csv('./data/data_mdd.csv')
mdd <- mdd[,c('eid','status','date_baseline')]
mdd <- mdd[(mdd$status==0)|((mdd$status==1)&(mdd$date_baseline<0)&(mdd$date_baseline>year)),]

##### Cov
cov <- read.csv('./data/data_cov.csv')
cov <- na.omit(cov)
cov$sex <- factor(cov$sex)
cov$smoking <- factor(cov$smoking)
cov$drinking <- factor(cov$drinking)
cov$education <- factor(cov$education)
cov$ethnic <- factor(cov$ethnic)

##### Omics
load(paste0("/public/home/xyhe/project/Metabologenomic_AD/DATA/Multiomics/",
            omic,"_imputed"))
if(exists("Proteome_imputed")){omics <- Proteome_imputed}
if(exists("Clinical_imputed")){omics <- Clinical_imputed}
if(exists("Metabolome_imputed")){omics <- Metabolome_imputed}
len_omics <- dim(omics)[2]-1

##### calculate logistic
data_all <- merge(mdd,cov,by='eid')

result <- data.frame(array(NA, dim=c(len_omics,7)))
names(result) <- c('Category','Phenotype','OR','z','P','N_total','N_mdd')

for(i in 1:len_omics){
  print(as.character(paste0(omic,': ',i)))
  pheno <- omics[,c(1,i+1)]
  pheno <- na.omit(pheno)
  pheno_name <- colnames(pheno)[2]
  colnames(pheno)[2] <- 'pheno'
  pheno$pheno <- scale(pheno$pheno, center = TRUE, scale = TRUE)
  data <- merge(pheno,data_all,by='eid')
  data <- na.omit(data)
  
  model <- glm(status ~ pheno + age_ins0 + sex + TDI + smoking + drinking + 
                 education + ethnic + bmi, data = data, family="binomial")
  x <- summary(model)
  x <- x$coefficients
  
  result[i,1] <- omic
  result[i,2] <- pheno_name
  result[i,3] <- exp(x["pheno","Estimate"])
  result[i,4] <- x["pheno","z value"]
  result[i,5] <- x["pheno","Pr(>|z|)"]
  result[i,6] <- as.numeric(dim(data)[1])
  result[i,7] <- as.numeric(dim(data[data$status==1,])[1])
 
}

write.csv(result,paste0(omic,'_before_',output_year[year_id],'.csv'), row.names=F)




