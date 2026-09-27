## R's randomForest, fit the way SL's forest expert is fit, over a seed sweep.
##
## Called by run.py with the design already written out, so both languages read
## byte-identical inputs and the only thing that varies is the forest.
suppressPackageStartupMessages(library(randomForest))

args <- commandArgs(trailingOnly = TRUE)
D        <- args[1]                       # working directory holding the inputs
design_f <- args[2]                       # design matrix, (T, p)
mtry     <- as.integer(args[3])           # -1 means R's own default, floor(p/3)
nodesize <- as.integer(args[4])
out_f    <- args[5]
nseed    <- as.integer(args[6])
ntrain   <- as.integer(args[7])

y <- as.numeric(readLines(file.path(D, "y.csv")))
design <- as.matrix(read.csv(file.path(D, design_f), header = FALSE))
colnames(design) <- paste0("V", seq_len(ncol(design)))
if (mtry < 0) mtry <- max(floor(ncol(design) / 3), 1)

train <- seq_len(ntrain)
datat <- as.data.frame(design[train, , drop = FALSE])
datap <- as.data.frame(design)

out <- matrix(NA_real_, nrow = nrow(design), ncol = nseed)
for (s in seq_len(nseed)) {
  set.seed(s)
  rf <- randomForest(x = datat, y = y[train], maxnodes = 20,
                     mtry = mtry, nodesize = nodesize, ntree = 500)
  out[, s] <- predict(rf, newdata = datap)
}
write.csv(out, file.path(D, out_f), row.names = FALSE)
cat(sprintf("%s p=%d mtry=%d nodesize=%d seeds=%d\n",
            out_f, ncol(design), mtry, nodesize, nseed))
