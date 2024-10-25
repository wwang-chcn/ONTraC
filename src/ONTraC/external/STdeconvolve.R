#! /usr/bin/Rscript

# ---------- Load modules ----------
library(STdeconvolve)

# ---------- parameters ----------
args = commandArgs(trailingOnly=TRUE)

counts_path <- args[1]
num_cell_type <- as.numeric(args[2])
save_directory <- args[3]

counts_df <- read.csv(counts_path, row.names = 1, header=TRUE)
counts_matrix <- as.matrix(counts_df)
sparse_counts <- as(counts_matrix, "dgCMatrix")

counts <- cleanCounts(sparse_counts, min.lib.size = 100, min.reads = 10)
corpus <- restrictCorpus(counts, removeAbove=1.0, removeBelow = 0.05, nTopOD = 1000)
ldas <- fitLDA(t(as.matrix(corpus)), Ks = c(num_cell_type))
optLDA <- optimalModel(models = ldas, opt = "min")
results <- getBetaTheta(optLDA, perc.filt = 0.05, betaScale = 1000)

write.table(results$theta, file = paste0(save_directory, "/spot_x_celltype_deconvolution.csv"), sep = ",")