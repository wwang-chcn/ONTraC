#! /usr/bin/Rscript

# ---------- Load modules ----------
library(STdeconvolve)

# ---------- parameters ----------
args <- commandArgs(trailingOnly = TRUE)

counts_path <- args[1]
num_cell_type <- as.numeric(args[2])
save_directory <- args[3]
spot_x_ct_name <- args[4]
ct_x_gene_name <- args[5]

counts_df <- read.csv(counts_path, row.names = 1, header = TRUE)
counts_df[is.na(counts_df)] <- 0
counts_matrix <- as.matrix(counts_df)
sparse_counts <- as(counts_matrix, "dgCMatrix")

if (nrow(sparse_counts) == 0 || ncol(sparse_counts) == 0) {
    stop("The input matrix is empty after filtering.")
}

counts <- cleanCounts(sparse_counts, min.lib.size = 50, min.reads = 10)
corpus <- restrictCorpus(
    counts,
    removeAbove = 1.0,
    removeBelow = 0.05,
    nTopOD = 1000
)
ldas <- fitLDA(t(as.matrix(corpus)), Ks = c(num_cell_type))
optLDA <- optimalModel(models = ldas, opt = "min") # nolint
results <- getBetaTheta(optLDA, perc.filt = 0.05, betaScale = 1000)

print(paste0("Saving deconvolved spot x cell type matrix to ", spot_x_ct_name))
write.table(
    results$theta,
    file = file.path(save_directory, spot_x_ct_name),
    sep = ","
)

if (!is.null(ct_x_gene_name)) {
    print(paste0(
        "Saving deconvolved cell type x gene matrix to ",
        ct_x_gene_name
    ))
    write.table(
        results$beta,
        file = file.path(save_directory, ct_x_gene_name),
        sep = ","
    )
}
