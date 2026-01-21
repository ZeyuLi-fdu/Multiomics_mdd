library(TwoSampleMR)
library(data.table)
library(ieugwasr)

##### Set
args <- commandArgs(trailingOnly = TRUE)
part <- as.numeric(args[1])
start <- (part-1)*500+1
end <- part*500

##### read outcome data
disease <- fread(file='mdd.txt')
disease <- data.frame(disease)
outcome_data <- format_data(disease,type="outcome",
                          snp_col = "ID",beta_col = "BETA",
                          se_col = "SE",effect_allele_col = "EA",
                          other_allele_col = "NEA",
                          pval_col = "PVAL",
                          chr_col = "X.CHROM",
                          phenotype_col = "phenotype")

##### read exposure data
protein_pqtl <- fread(file='cispQTL_IV_UKB.txt')
protein_pqtl <- data.frame(protein_pqtl)
protein_list <- unique(protein_pqtl$Pro_code)
protein_list <- protein_list[order(protein_list)]

for (i in start:end){
  protein <- protein_list[i]
  exposure_data <- protein_pqtl[protein_pqtl$Pro_code==protein,]
  exposure_data$phenotype <- protein
  exposure_data <- format_data(exposure_data,type="exposure",
                             snp_col = "rsid",beta_col = "BETA",
                             se_col = "SE",effect_allele_col = "ALLELE1",
                             other_allele_col = "ALLELE0",
                             pval_col = "P",
                             chr_col = "chr",
                             phenotype_col = "phenotype")
  ##### MR analysis
  dat <- harmonise_data(exposure_dat = exposure_data, outcome_dat = outcome_data)
  dat_t <- dat[dat$mr_keep=='TRUE',]
  if(as.numeric(dim(dat_t)[1])<1) {next}
  res <- mr(dat)
  res <- generate_odds_ratios(res)

  if(exists("res")) { 
    if(!exists("res_final")) {res_final<-res} else {res_final <- rbind(res_final,res)} 
    }
}
