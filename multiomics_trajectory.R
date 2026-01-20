pacman::p_load(data.table,dplyr,plyr,magrittr,MatchIt,tidyr,Mfuzz,cowplot,ggplot2,tibble,purrr,stringr,DEswan)

endpoint <- 'MDD'
span <- 0.8
omics <- c('Metabolome', 'Proteome', 'Clinical')

results_dir <- './results/multiomics_trajectory/'
dir.create(results_dir, recursive = TRUE)

load('BloodDict_detailed.RData')

outcome_status <- paste0(endpoint, '_status')
outcome_years <- paste0(endpoint, '_years')

outcome <- fread('data_mdd.csv') %>% as.data.frame() %>%
  select(eid, status, date_baseline) %>%
  rename(!!sym(outcome_status) := status, !!sym(outcome_years) := date_baseline)

multiomics_id <- fread('stage2_id.csv') %>% as.data.frame() %>% pull(eid)

Covar <- fread('data_cov.csv') %>% as.data.frame() %>%
  rename(age = age_ins0, tdi = TDI, smk = smoking, College = education, ethn = ethnic)

covariates <- c("age", "sex", "bmi")

for (omic in omics) {
  variable.names <- BloodDict$Omics_feature[BloodDict$Omics_group == omic]
  
  omic_data <- get(load(paste0(omic, "_imputed.RData")))
  
  data_merged <- omic_data %>% 
    inner_join(Covar, by = 'eid') %>% 
    inner_join(outcome, by = 'eid') %>% 
    filter(eid %in% multiomics_id) %>% 
    na.omit() %>% 
    filter(.data[[outcome_years]] > 0)
  
  case <- data_merged %>% filter(.data[[outcome_status]] == 1)
  control <- data_merged %>% filter(.data[[outcome_status]] == 0)
  match_final <- rbind(case, control)
  
  mt_out1 <- matchit(as.formula(paste(outcome_status, "~ age + College + bmi")),
                     method = "nearest", distance = "mahalanobis", ratio = 10, 
                     link = "logit", exact = ~sex, data = match_final)
  
  mt_data <- match.data(mt_out1)
  mt_final <- mt_data %>% arrange(subclass)
  
  time_scale <- mt_final[[outcome_years]][which(mt_final[[outcome_status]] == 1)]
  subclass_sizes <- mt_final %>% dplyr::group_by(subclass) %>% dplyr::summarise(n = n())
  repeated_time_scale <- rep(time_scale, times = subclass_sizes$n)
  mt_final <- mt_final %>% mutate(time_scale = repeated_time_scale * -1)
  
  resid_file <- file.path(results_dir, paste0("AC_resid_", endpoint, '_', omic, ".RData"))
  if (file.exists(resid_file)) {
    load(resid_file)
  } else {
    resid_mt <- mt_final %>% magrittr::set_rownames(.$eid)
    for (i in variable.names) {
      FML <- paste0(i, " ~ ", paste(covariates, collapse = ' + '))
      sub_data <- resid_mt[, c('eid', i, covariates)] %>% na.omit() %>% magrittr::set_rownames(.$eid)
      sub_data[, i] <- resid(lm(as.formula(FML), data = sub_data))
      resid_mt[, i] <- NA
      resid_mt[rownames(sub_data), i] <- as.numeric(scale(sub_data[, i]))
    }
    save(resid_mt, file = resid_file)
  }
  
  zscore_file <- file.path(results_dir, paste0("AC_trajectories_Zscore_", endpoint, '_', omic, ".RData"))
  if (file.exists(zscore_file)) {
    load(zscore_file)
  } else {
    df_group_case <- resid_mt %>% ungroup() %>% filter(.data[[outcome_status]] == 1)
    df_group_control <- resid_mt %>% ungroup() %>% filter(.data[[outcome_status]] == 0)
    
    stats_control <- df_group_control %>%
      summarise_at(vars(all_of(variable.names)),
                   list(mean = ~mean(., na.rm = TRUE), sd = ~sd(., na.rm = TRUE))) %>%
      tidyr::pivot_longer(cols = everything(), names_to = c("Omics_feature", ".value"), 
                          names_pattern = "(.*)_(mean|sd)") %>%
      as.data.frame() %>% set_rownames(.$Omics_feature)
    
    df_z_scores <- df_group_case
    for (var_name in variable.names) {
      df_z_scores[[var_name]] <- (df_group_case[[var_name]] - stats_control[var_name, 'mean']) / stats_control[var_name, 'sd']
    }
    
    t <- df_z_scores %>% dplyr::select(eid, subclass, Year = time_scale, all_of(variable.names))
    save(t, file = zscore_file)
  }
  
  load(zscore_file)
  df_t <- t %>% tidyr::pivot_longer(cols = all_of(variable.names), 
                                    names_to = "Characteristics", values_to = 'Estimate')
  
  df_loess <- do.call(rbind, lapply(variable.names, function(var_name) {
    df2 <- df_t %>% filter(Characteristics == var_name) %>% na.omit()
    loess_fit <- loess(Estimate ~ Year, data = df2, span = span)
    gap <- seq(-14.5, -0.5, 0.5)
    data.frame(Year = gap, Characteristics = rep(var_name, length(gap))) %>%
      dplyr::mutate(Estimate_loess = predict(loess_fit, newdata = .))
  }))
  
  loess_dir <- file.path(results_dir, paste0("loess_span", span))
  dir.create(loess_dir, recursive = TRUE)
  save(df_loess, file = file.path(loess_dir, paste0("AC_trajectories_", endpoint, '_', omic, ".RData")))
}

# Trajectory cluster

endpoint <- 'MDD'
span <- 0.8
optimal_clusters <- 7
membership_cutoff <- 0.5

results_dir <- './results/multiomics_trajectory/'
loess_path <- file.path(results_dir, paste0("loess_span", span))
cluster_path <- file.path(loess_path, 'Mfuzz_Cluster/')
dir.create(cluster_path, recursive = TRUE)

omics_color <- c('Metabolome' = "#BC80BD", 'Proteome' = "#8DD3C7", 'Clinical' = "#FDB462")

load('BloodDict_detailed.RData')

sig_files <- list.files('./association_result/', pattern = "_.*Sig_in_Stage1\\.csv", full.names = TRUE)
sig_features <- lapply(sig_files, function(file) fread(file)$Phenotype)
sig_features_combined <- unique(unlist(sig_features))

omics <- c('Metabolome', 'Proteome', 'Clinical')
loess_omics <- list()
for (omic in omics) {
  load(file.path(loess_path, paste0("AC_trajectories_", endpoint, '_', omic, ".RData")))
  df_loess <- df_loess %>% filter(Characteristics %in% BloodDict$Omics_feature[BloodDict$Omics_group == omic])
  loess_omics[[omic]] <- df_loess
}
loess_omics <- do.call(plyr::rbind.fill, loess_omics)

df_wide <- loess_omics %>% 
  filter(Characteristics %in% sig_features_combined) %>%
  pivot_wider(names_from = Characteristics, values_from = Estimate_loess) %>%
  column_to_rownames("Year") %>% 
  t() %>% 
  as.matrix()

mfuzz_class <- new('ExpressionSet', exprs = df_wide)
mfuzz_class_scaled <- standardise(mfuzz_class)

m1 <- mestimate(mfuzz_class_scaled)

set.seed(123)
cluster_subpath <- file.path(cluster_path, endpoint)
dir.create(cluster_subpath, recursive = TRUE)

pdf(file.path(cluster_subpath, paste0(endpoint, '.pdf')), width = 5, height = 5)
Dmin(mfuzz_class_scaled, m = m1, crange = 2:15, repeats = 3, visu = TRUE)
dev.off()

fcm_final <- mfuzz(mfuzz_class_scaled, c = optimal_clusters, m = m1)

center <- get_mfuzz_center(data = mfuzz_class_scaled, c = fcm_final, membership_cutoff = membership_cutoff)
rownames(center) <- paste("Cluster", rownames(center), sep = ' ')

cluster_info <- data.frame(
  Omics_feature = names(fcm_final$cluster),
  fcm_final$membership,
  cluster = fcm_final$cluster,
  stringsAsFactors = FALSE
) %>% arrange(cluster)

cluster_specific <- cluster_center <- plot_idx <- list()
for (idx in 1:optimal_clusters) {
  cluster_data <- cluster_info %>% 
    filter(cluster == idx) %>% 
    select(Omics_feature, all_of(paste0("X", idx)), cluster) %>%
    rename(membership = 2) %>%
    filter(membership >= membership_cutoff)
  
  cluster_center[[idx]] <- center[idx, , drop = FALSE] %>% 
    t() %>% 
    as.data.frame() %>% 
    tibble::rownames_to_column("time") %>%
    rename(value = 1) %>%
    mutate(cluster = idx, time = as.numeric(time))
  
  cluster_specific[[idx]] <- mfuzz_class_scaled@exprs[cluster_data$Omics_feature, , drop = FALSE] %>%
    as.data.frame() %>%
    tibble::rownames_to_column("Omics_feature") %>%
    mutate(cluster = cluster_data$cluster[match(Omics_feature, cluster_data$Omics_feature)],
           membership = cluster_data$membership[match(Omics_feature, cluster_data$Omics_feature)]) %>%
    pivot_longer(cols = -c(Omics_feature, cluster, membership), names_to = "time", values_to = "value") %>%
    mutate(time = as.numeric(time)) %>%
    left_join(BloodDict[, c("Omics_feature", "Omics_group")], by = "Omics_feature") %>%
    arrange(desc(membership), Omics_feature) %>%
    arrange(desc(Omics_group)) %>%
    mutate(Omics_feature = factor(Omics_feature, levels = unique(Omics_feature)))
  
  plot_idx[[idx]] <- ggplot(cluster_specific[[idx]], aes(time, value, group = Omics_feature)) +
    geom_line(aes(color = Omics_group), alpha = 0.7) +
    geom_line(data = cluster_center[[idx]], aes(time, value), size = 2, inherit.aes = FALSE) +
    geom_hline(yintercept = 0) +
    scale_color_manual(values = omics_color) +
    theme_bw() +
    theme(
      legend.position = 'none',
      panel.grid = element_blank(),
      axis.title = element_text(size = 13),
      axis.text = element_text(size = 12),
      panel.background = element_blank(),
      plot.background = element_blank()
    ) +
    labs(x = "", y = "Z-score", title = paste("Cluster ", idx, " (", nrow(cluster_data), " molecules)"))
}

plot_cluster <- do.call(plot_grid, c(plot_idx, align = 'hv', ncol = 3))

single_width <- 3
single_height <- 3
ncol_plot <- 3
nrow_plot <- ceiling(optimal_clusters / ncol_plot)
ggsave(filename = file.path(cluster_subpath, paste0("Cluster_", optimal_clusters, ".pdf")),
       plot = plot_cluster, width = single_width * ncol_plot, height = single_height * nrow_plot, units = "in")

cluster_info <- unique(cluster_info$cluster) %>% 
  purrr::map(function(x) {
    temp <- cluster_info %>% 
      select(Omics_feature, paste0("X", x), cluster) %>%
      rename(membership = 2) %>%
      filter(membership >= membership_cutoff)
    temp
  }) %>% 
  bind_rows()

save(cluster_info, cluster_specific, cluster_center, 
     file = file.path(cluster_subpath, paste0("Cluster_", optimal_clusters, ".RData")))
	 
# DESWAN

endpoint <- 'MDD'
results_dir <- './results/multiomics_trajectory/deswan/'
load('BloodDict_detailed.RData')

deswan_c <- c("age", "sex", "bmi")
outcome_status <- paste0(endpoint, '_status')
outcome_years <- paste0(endpoint, '_years')

omics <- c('Metabolome', 'Proteome', 'Clinical')

outcome <- fread('data_mdd.csv') %>% as.data.frame() %>%
  select(eid, !!sym(outcome_status) := status, !!sym(outcome_years) := date_baseline)

multiomics_id <- fread('stage2_id.csv') %>% as.data.frame() %>% pull(eid)

Covar <- fread('data_cov.csv') %>% as.data.frame() %>%
  rename(age = age_ins0, tdi = TDI, smk = smoking, College = education, ethn = ethnic)

omic_data <- list()
for (omic in omics) {
  omic_features <- BloodDict$Omics_feature[BloodDict$Omics_group == omic]
  omic_data[[omic]] <- get(load(paste0(omic, "_imputed.RData"))) %>%
    select(eid, all_of(omic_features)) %>%
    filter(eid %in% multiomics_id)
}

omics_dat <- Reduce(function(x, y) merge(x, y, by = 'eid'), omic_data) %>% 
  as.data.frame() %>%
  inner_join(Covar, by = 'eid') %>%
  inner_join(outcome, by = 'eid') %>%
  filter(.data[[outcome_status]] == 1) %>%
  select(eid, Year = all_of(outcome_years), all_of(BloodDict$Omics_feature), all_of(deswan_c)) %>%
  mutate(Year = -Year)

res_DEswan <- res_p <- list()
for (parcel_width in 1:9) {
  character_parcel_width <- as.character(parcel_width)
  res_DEswan[[character_parcel_width]] <- DEswan(
    data.df = omics_dat[, BloodDict$Omics_feature],
    qt = omics_dat[, 'Year'],
    covariates = omics_dat[, deswan_c],
    window.center = seq(-13, -2, 1),
    buckets.size = parcel_width
  )
  res_p[[character_parcel_width]] <- reshape.DEswan(res_DEswan[[character_parcel_width]], parameter = 1, factor = "qt")
}

save(res_DEswan, res_p, file = file.path(results_dir, paste0(endpoint, ".RData")))