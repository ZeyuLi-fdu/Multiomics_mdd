library(data.table)
library(survival)

##### Set
args <- commandArgs(trailingOnly = TRUE)
stage <- as.numeric(args[1])
omic_id <- as.numeric(args[2])

stage_eid <- read.csv(paste0("./association/stage",stage,"_id.csv"))
name_omics <- c('Proteome','Clinical','Metabolome')
omic <- name_omics[omic_id]

##### MDD
mdd <- read.csv('./data/data_mdd.csv')
mdd <- mdd[,c('eid','status','date_baseline')]
mdd <- mdd[mdd$date_baseline>0,]

##### Cov
cov <- read.csv('./data/data_cov.csv')
cov <- na.omit(cov)
cov$sex <- factor(cov$sex)
cov$smoking <- factor(cov$smoking)
cov$drinking <- factor(cov$drinking)
cov$education <- factor(cov$education)
cov$ethnic <- factor(cov$ethnic)

##### Omics
load(paste0(omic,"_imputed"))
if(exists("Proteome_imputed")){omics <- Proteome_imputed}
if(exists("Clinical_imputed")){omics <- Clinical_imputed}
if(exists("Metabolome_imputed")){omics <- Metabolome_imputed}
len_omics <- dim(omics)[2]-1

##### calculate cox
data_all <- merge(stage_eid,mdd,by='eid')
data_all <- merge(data_all,cov,by='eid')

result <- data.frame(array(NA, dim=c(len_omics,9)))
names(result) <- c('Category','Phenotype','HR','z','P','N_total','N_mdd','Chisq','Chisq_P')

for(i in 1:len_omics){
  print(as.character(paste0(omic,': ',i)))
  pheno <- omics[,c(1,i+1)]
  pheno <- na.omit(pheno)
  pheno_name <- colnames(pheno)[2]
  colnames(pheno)[2] <- 'pheno'
  pheno$pheno <- scale(pheno$pheno, center = TRUE, scale = TRUE)
  data <- merge(pheno,data_all,by='eid')
  data <- na.omit(data)
  
  model <- coxph(Surv(date_baseline, status) ~ pheno + age_ins0 + sex + TDI + smoking +
                 drinking + education + ethnic + bmi, data = data)
  x <- summary(model)
  x <- x$coefficients
  PH_test <- cox.zph(model)
  PH_test <- PH_test$table
  
  result[i,1] <- omic
  result[i,2] <- pheno_name
  result[i,3] <- x["pheno","exp(coef)"]
  result[i,4] <- x["pheno","z"]
  result[i,5] <- x["pheno","Pr(>|z|)"]
  result[i,6] <- as.numeric(dim(data)[1])
  result[i,7] <- as.numeric(dim(data[data$status==1,])[1])
  result[i,8] <- PH_test["pheno","chisq"]
  result[i,9] <- PH_test["pheno","p"]
 
}

write.csv(result,paste0(omic,'_Stage',stage,'.csv'),row.names = F)




