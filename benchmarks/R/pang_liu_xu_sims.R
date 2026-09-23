#!/usr/bin/env Rscript
# Pang, Liu & Xu (2022) "A Bayesian Alternative to Synthetic Control for
# Comparative Case Studies", Political Analysis 30(2):269-288 -- the R-side
# reference for the single-treated-unit simulations of Appendix Tables A6 and
# A7.
#
# Replication package: Harvard Dataverse doi:10.7910/DVN/B6SWA1. The two
# designs are the archive's 9_sim_single_r8.R (eight weak factors, no
# covariates) and 10_sim_single_X.R (three strong factors plus six
# time-invariant covariates with time-varying coefficients). Both draw panels
# from the archive's code/simulateCalib.R and summarise the sampler with
# effSummary from code/summary_function.R; both functions are reproduced below
# verbatim so this script runs on its own.
#
# The treatment effect these designs generate is identically zero -- the block
# that would have written a non-zero effect is commented out in simulateCalib.R
# and the drivers read true.eff off the same all-zero column. So the tables are
# placebo studies: their "bias" column is the mean estimate, and their coverage
# column is coverage of zero.
#
# Writes, into --out:
#   seam_panels_<design>.csv  the panels, long, one block per (design, case)
#   seam_reference.csv  gsynth's and pblasso's output on each of those panels
#   dgp_moments.csv     moments of the generator, averaged over --moment-draws
#   effect_column.csv   the largest effect simulateCalib.R writes, over
#                       --effect-draws panels, which is how the zero effect is
#                       established and not assumed
#   ar1_moments.csv     second moments of arima.sim itself, at the three path
#                       lengths the designs use
#
# --only takes moments, seam, effect, arsim or all (the default), so one part
# can be regenerated without redoing the others.
#
# ar1_moments.csv exists because the panel moments cannot see the one
# substitution a Python transcription has to make. arima.sim starts the
# recursion at its stationary distribution; starting it at zero instead costs
# about 2 per cent of E[x^2] at length 30, and by the time that has been
# diluted by the loadings, the covariates and an error standard deviation of 5
# it moves the cross-sectional moment by about 1 per cent -- inside the noise
# of any affordable number of panel draws. Measured on the series alone it is
# a fortyfold separation.
#
# The Python case reads all of them. The panel files and seam_reference.csv let
# it put mlsynth's GSYNTH and DMLFM on the same panels the R arms saw;
# dgp_moments.csv pins its own transcription of the generator.

suppressMessages({
  library(gsynth)
  library(pblasso)
})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) default else args[i + 1]
}
out.dir <- getarg("--out", "reference")
moment.draws <- as.integer(getarg("--moment-draws", "2000"))
effect.draws <- as.integer(getarg("--effect-draws", "500"))
only <- getarg("--only", "all")
wants <- function(part) only %in% c("all", part)
seed <- as.integer(getarg("--seed", "20220301"))
dir.create(out.dir, showWarnings = FALSE, recursive = TRUE)

## ------------------------------------------------------------------ ##
## The authors' generator, code/simulateCalib.R, verbatim.
## ------------------------------------------------------------------ ##

getTS <- function(type = "ar1", TT, ar.coef) {
      if (type %in% c("ar1","drift")) {
        ts <- arima.sim(list(order = c(1,0,0), ar = ar.coef), n = TT)
        } else {
        ts <- rnorm(TT)
        }
        if (type == "drift") {
           ts <- ts + 0.5*seq(from=0, length.out = TT, by=1)
        }
        return(ts)
}

simulateCalib<-function(
    N,
    TT,
    tr.threshold = 0.5,
    tr.start = 10,
    tr.coef = NULL,
    p = 10,
    beta = c(4,4,2,2,rep(0, 6)),
    time.invariant = FALSE,
    alpha = NULL,
    xi = NULL,
    mu=0,
    force,
    time.eff = NULL,
    alpha.sd = 1,
    lambda.mean = NULL,
    lambda.sd = NULL,
    factors = NULL,
    Rtype = "n",
    error.type = "n",
    error.sd = 1,
    tr.noise = 0.5,
    ar.coef = NULL,
    seed=NULL
    ) {

    if (is.null(seed)==FALSE) {set.seed(seed)}

    ntr <- length(tr.threshold) + 1

    r <- length(lambda.sd)
    lambda <- matrix(NA, N, r)
    for (i in 1:r) {
        lambda[,i] <- rnorm(N, lambda.mean[i], lambda.sd[i])
    }
    if (is.null(factors)==TRUE) {
        factors <- matrix(NA, TT, r)
        for (i in 1:r) {
            factors[,i] <- getTS(type = "ar1", TT, ar.coef)
        }
    }

    if (p>0) {

        if (time.invariant == FALSE) {
            X <- array(rnorm(N*TT*p), dim = c(TT, N, p))
        } else {
            W <- matrix(rnorm(N*p), nrow = N, ncol = p)
            X <- array(NA, dim = c(TT, N, p))
            for (i in 1:TT) {
                X[i,,] <- W
            }
        }
        Xfit <- matrix(0, TT, N)
        for (i in 1:p) {
            Xfit <- Xfit + X[,,i] * beta[i]
        }
        Zfit <- matrix(0, TT, N)
        if (is.null(alpha)==FALSE) {
            for (i in 1:p) {
                Zfit <- Zfit + X[,,i] * matrix(rep(alpha[,i], each = TT), TT, N)
            }
        } else {
            alpha <- matrix(0, N, p)
        }
        if (is.null(xi)==TRUE) {
            xi <- matrix(0, TT, p)
            for (i in c(1:p)) {
                xi[,i] <- getTS(type = "ar1", TT, ar.coef)
            }
        }
        Afit <- matrix(0, TT, N)
        for (i in 1:p) {
            Afit <- Afit + X[,,i] * matrix(rep(xi[,i], N), TT, N)
        }
    }

    if (force==1|force==3) {
        unit.fe <- rnorm(N,sd=alpha.sd)
    } else {
        unit.fe <- rep(0, N)
    }
    unitFE <- matrix(rep(unit.fe,each=TT),TT,N)
    if (force==2|force==3) {
        if (is.null(time.eff)==TRUE) {
            ts <- arima.sim(list(order = c(1,0,0), ar = ar.coef), TT)
        } else {
            ts <- time.eff
        }
        timeFE <- matrix(rep(ts,N),TT,N)
    }

    if (r == 1) {
        ps.raw <- tr.coef[1]*lambda[,1] + tr.coef[2]* unit.fe + rnorm(N, 0, tr.noise)
    } else {
        ps.raw <- tr.coef[1]*lambda[,1] + tr.coef[2]*lambda[,2] + tr.coef[3]* unit.fe + rnorm(N, 0, tr.noise)
    }
    if (p > 0 & time.invariant == TRUE) {
        ps.raw <- ps.raw + W[,1] * tr.coef[4] + W[,2]* tr.coef[5]
    }
    tr.star <- (ps.raw - min(ps.raw))/(max(ps.raw)-min(ps.raw))
    tr.rank <- rank(tr.star)
    tr.rank <- (tr.rank-1)/(max(tr.rank)-1)
    D <- matrix(0, TT, N)
    treat <- rep(0, N)
    T0 <- rep(TT, N)
    for (i in 1:(ntr-1)) {
        tr.id <- which(tr.rank*10000>=tr.threshold[i]*10000+1)
        treat[tr.id] <- i
        T0[tr.id] <- tr.start[i]-1
        for (j in tr.id) {
            D[tr.start[i]:TT,j] <- 1
        }
    }

    e <- rnorm(TT*N,sd=error.sd)
    e <- matrix(e, TT, N)

    Y <- matrix(mu, TT, N)

    if (r>0) {
        Y <- Y + factors%*%t(lambda)
    }
    if (p>0) {
        Y <- Y + Xfit + Zfit + Afit
    }

    if (force==1|force==3) {
        Y <- Y + 1*unitFE
    }
    if (force==2|force==3) {
        Y <- Y + 1*timeFE
    }

    eff <- matrix(0,TT,N)
    Y <- Y + eff

    Y <- Y + e

    panel<-as.data.frame(cbind(
        rep(101:(100+N),each=TT),
        rep(1:TT,N),
        rep(1:TT,N),
        c(Y),
        c(e),
        c(eff),
        rep(T0, each = TT)))
    cname<-c("id","time","t","Y","error","eff","T0")

    treat <- rep(D[nrow(D),], each = TT)
    panel <- cbind(panel,c(D), treat)
    cname <- c(cname,"D","treat")

    if (p>0) {
        for (i in 1:p) {
            panel<-cbind(panel,c(X[,,i]))
            cname<-c(cname,paste("X",i,sep=""))
        }
    }

    if (force==1|force==3) {
        panel <- cbind(panel,c(unitFE))
        cname <- c(cname,"unitFE")
    }
    if (force==2|force==3) {

        panel <- cbind(panel,c(timeFE))
        cname <- c(cname,"timeFE")
    }

    if (r>0) {
        for (i in 1:r) {
            panel<-cbind(panel,rep(factors[,i],N))
            cname<-c(cname,paste("F",i,sep=""))
        }
        for (i in 1:r) {
            panel<-cbind(panel,rep(lambda[,i],each=TT))
            cname<-c(cname,paste("L",i,sep=""))
        }
    }
    colnames(panel)<-cname

    if (r>0) for (i in 1:r) {
        panel[,paste("FL",i,sep="")]<-panel[,paste("F",i,sep="")]*panel[,paste("L",i,sep="")]
    }

    return(panel)
}

## ------------------------------------------------------------------ ##
## The authors' effSummary, code/summary_function.R, verbatim.
## ------------------------------------------------------------------ ##

effSummary <- function(x,
                       usr.id = NULL,
                       burn = 1000,
                       cumu = FALSE,
                       expo = FALSE,
                       rela.period = TRUE) {

    niter <- dim(x$sigma2_i)[2]

    if (cumu) {
        rela.period <- TRUE
    }

    id.tr <- x$raw.id.tr
    time.tr <- x$time.tr
    rela.time.tr <- x$rela.time.tr

    id.pos <- NULL
    unique.tr <- c(unique(id.tr))
    if (is.null(usr.id)) {
        id.pos <- 1:length(c(id.tr))
    } else {
        if (sum(usr.id %in% unique.tr) != length(usr.id)) {
            stop("Some specified ids are not in treated group, please check input.\n")
        }
        id.pos <- which(c(id.tr) %in% usr.id)
    }

    yo_t <- NULL
    if (expo) {
        yo_t <- exp(x$yo_t)
    } else {
        yo_t <- x$yo_t
    }
    yo_t <- yo_t[id.pos]

    time.tr <- time.tr[id.pos]
    rela.time.tr <- rela.time.tr[id.pos]

    yct_i <- NULL
    if (expo) {
        yct_i <- exp(x$yct)
    } else {
        yct_i <- x$yct
    }

    yct_i <- matrix(c(yct_i[id.pos, (burn + 1):niter]), length(id.pos), niter - burn)

    count.tr <- NULL

    if (rela.period) {
        m_yo <- tapply(yo_t, rela.time.tr, mean)
        m_yct <- sapply(1:(niter - burn), function(i){tapply(yct_i[, i], rela.time.tr, mean)})
        count.tr <- as.numeric(table(rela.time.tr))
    } else {
        m_yo <- tapply(yo_t, time.tr, mean)
        m_yct <- sapply(1:(niter - burn), function(i){tapply(yct_i[, i], time.tr, mean)})

        count.tr <- as.numeric(table(rela.time.tr))
    }

    m_yct_mean <- apply(m_yct, 1, mean)
    m_yct_ci_l <- apply(m_yct, 1, quantile, 0.025)
    m_yct_ci_u <- apply(m_yct, 1, quantile, 0.975)

    eff_i <- matrix(rep(c(m_yo), niter - burn), length(c(m_yo)), niter - burn) - m_yct

    eff_mean <- apply(eff_i, 1, mean)
    eff_ci_l <- apply(eff_i, 1, quantile, 0.025)
    eff_ci_u <- apply(eff_i, 1, quantile, 0.975)

    data <- cbind.data.frame(m_yo, m_yct_mean, m_yct_ci_u, m_yct_ci_l, eff_mean, eff_ci_l, eff_ci_u)
    names(data) <- c("observed", "estimated_counterfactual",
                     "counterfactual_ci_l", "counterfactual_ci_u",
                     "estimated_ATT", "estimated_ATT_ci_l", "estimated_ATT_ci_u")
    if(rela.period) {
        data$time <- sort(unique(rela.time.tr))
        data$count <- count.tr
    } else {
        data$time <- sort(unique(time.tr))
    }

    est.eff <- data

    t.post <- which(rela.time.tr > 0)

    eff_avg_i <- sapply(1:(niter - burn), function(i) {mean(yo_t[t.post] - yct_i[t.post, i])})

    eff_avg_mean <- mean(eff_avg_i)
    eff_avg_ci_l <- quantile(eff_avg_i, 0.025)
    eff_avg_ci_u <- quantile(eff_avg_i, 0.975)

    est.avg <- cbind(eff_avg_mean, eff_avg_ci_l, eff_avg_ci_u)
    colnames(est.avg) <- c("mean", "ci_l", "ci_u")

    out <- list(est.eff = est.eff,
                est.avg = est.avg)

    return(out)
}

## ------------------------------------------------------------------ ##
## The two designs, as their drivers set them.
## ------------------------------------------------------------------ ##

# 9_sim_single_r8.R: p = 0, r = 8, lambda.sd = 2, gsynth is given r = 8.
# 10_sim_single_X.R: p = 6 time-invariant covariates with beta = (4,3,2,1,0,0)
#   and per-replication AR(1) xi scaled by beta, r = 3, lambda.sd = 4, and
#   gsynth is given r + 4 = 7 -- not the true rank, because the covariate
#   terms carry rank of their own.
# Both: force = 2 (time effects only), error.sd = 5, ar.coef = 0.6, ten post
# periods, and a time effect drawn once per case and held fixed.

draw <- function(design, N, TT, T0, time.eff) {
  if (design == "r8") {
    simulateCalib(N = N, TT = TT, tr.threshold = (N - 1) / N, tr.start = T0 + 1,
                  tr.coef = c(0.1, 0.1, 0.1), tr.noise = 1, p = 0, mu = 0,
                  force = 2, time.eff = time.eff, lambda.mean = rep(0, 8),
                  lambda.sd = rep(2, 8), factors = NULL, error.sd = 5, ar.coef = 0.6)
  } else {
    beta <- c(4, 3, 2, 1, 0, 0)
    xi <- matrix(0, TT, 6)
    for (i in 1:6) xi[, i] <- getTS(type = "ar1", TT, 0.6) * beta[i]
    simulateCalib(N = N, TT = TT, tr.threshold = (N - 1) / N, tr.start = T0 + 1,
                  tr.coef = rep(0.1, 5), tr.noise = 1, p = 6, beta = beta, xi = xi,
                  time.invariant = TRUE, mu = 0, force = 2, time.eff = time.eff,
                  lambda.mean = rep(0, 3), lambda.sd = rep(4, 3), factors = NULL,
                  error.sd = 5, ar.coef = 0.6)
  }
}

GSYNTH_R <- c(r8 = 8, X = 7)

## ------------------------------------------------------------------ ##
## 1. Moments of the generator.
## ------------------------------------------------------------------ ##
# The time effect is redrawn here, which the drivers do not do. Held fixed, a
# single draw of a series whose deterministic part climbs by four a period
# decides most of the within-unit variance, and the moment would report that
# draw and not the generator.

set.seed(seed)
configs <- list(list("r8", 31, 30, 20), list("r8", 51, 70, 60),
                list("X", 31, 30, 20), list("X", 51, 70, 60))
mom.rows <- list()
for (cf in if (wants("moments")) configs else list()) {
  design <- cf[[1]]; N <- cf[[2]]; TT <- cf[[3]]; T0 <- cf[[4]]
  acc <- matrix(NA, moment.draws, 6)
  for (k in 1:moment.draws) {
    time.eff <- getTS(type = "drift", TT, 0.6) * 8
    d <- draw(design, N, TT, T0, time.eff)
    Ym <- matrix(d$Y, TT, N)
    trcol <- unique(d$id[d$D == 1]) - 100
    dm <- Ym - rowMeans(Ym)
    acc[k, ] <- c(var(c(Ym)), mean(apply(Ym, 1, var)), var(Ym[1:T0, trcol]),
                  mean(apply(dm, 2, var)), var(dm[, trcol]), mean(Ym[, trcol]))
  }
  mom.rows[[length(mom.rows) + 1]] <- data.frame(
    design = design, N = N, TT = TT, T0 = T0, draws = moment.draws,
    var_y = mean(acc[, 1]), var_within_time = mean(acc[, 2]),
    var_treated_pre = mean(acc[, 3]), var_within_unit_demeaned = mean(acc[, 4]),
    var_treated_demeaned = mean(acc[, 5]), mean_treated = mean(acc[, 6]))
  cat(sprintf("moments %-2s N=%2d TT=%2d done\n", design, N, TT))
}
if (wants("moments")) {
  write.csv(do.call(rbind, mom.rows), file.path(out.dir, "dgp_moments.csv"), row.names = FALSE)
}

## ------------------------------------------------------------------ ##
## 1b. The effect column.
## ------------------------------------------------------------------ ##
# simulateCalib.R writes eff as a zero matrix and the loop that would fill it
# is commented out, so both drivers read a true effect of zero off it. This
# measures that on the authors' own code instead of taking it from a reading.

if (wants("effect")) {
  set.seed(seed + 7)
  eff.rows <- list()
  for (cf in configs) {
    design <- cf[[1]]; N <- cf[[2]]; TT <- cf[[3]]; T0 <- cf[[4]]
    worst <- 0; worst.tr <- 0; ntr <- integer(0)
    for (k in 1:effect.draws) {
      d <- draw(design, N, TT, T0, getTS(type = "drift", TT, 0.6) * 8)
      worst <- max(worst, max(abs(d$eff)))
      worst.tr <- max(worst.tr, max(abs(d$eff[d$D == 1])))
      ntr <- c(ntr, length(unique(d$id[d$D == 1])))
    }
    eff.rows[[length(eff.rows) + 1]] <- data.frame(
      design = design, N = N, TT = TT, T0 = T0, draws = effect.draws,
      max_abs_eff = worst, max_abs_eff_treated = worst.tr,
      min_treated_units = min(ntr), max_treated_units = max(ntr))
    cat(sprintf("effect %-2s N=%2d TT=%2d done\n", design, N, TT))
  }
  write.csv(do.call(rbind, eff.rows), file.path(out.dir, "effect_column.csv"),
            row.names = FALSE)
}

## ------------------------------------------------------------------ ##
## 1c. The AR(1) generator on its own.
## ------------------------------------------------------------------ ##

if (wants("arsim")) {
  set.seed(seed + 13)
  ar.rows <- list()
  for (TT in c(30, 50, 70)) {
    v <- replicate(20000, {
      x <- arima.sim(list(order = c(1, 0, 0), ar = 0.6), n = TT)
      c(mean(x ^ 2), var(x))
    })
    ar.rows[[length(ar.rows) + 1]] <- data.frame(
      TT = TT, draws = 20000, ar_coef = 0.6,
      mean_x2 = mean(v[1, ]), mean_var = mean(v[2, ]))
    cat(sprintf("arsim T=%2d done\n", TT))
  }
  write.csv(do.call(rbind, ar.rows), file.path(out.dir, "ar1_moments.csv"),
            row.names = FALSE)
}

## ------------------------------------------------------------------ ##
## 2. Seam panels and the R arms on them.
## ------------------------------------------------------------------ ##
# pblasso runs at the drivers' niter = 10000 / burn = 2000 and again at the
# reduced 1500 / 375 the Python case can afford, so the cost of the reduction
# is measured here and not assumed.

set.seed(seed + 1)
seam.configs <- list(list("r8", 31, 30, 20), list("r8", 51, 50, 40),
                     list("X", 31, 30, 20), list("X", 51, 50, 40))
panels <- list(r8 = list(), X = list()); refs <- list()
for (cf in if (wants("seam")) seam.configs else list()) {
  design <- cf[[1]]; N <- cf[[2]]; TT <- cf[[3]]; T0 <- cf[[4]]
  time.eff <- getTS(type = "drift", TT, 0.6) * 8
  for (rep in 1:2) {
    tag <- sprintf("%s_N%d_T%d_rep%d", design, N - 1, T0, rep)
    d <- draw(design, N, TT, T0, time.eff)
    keep <- c("id", "time", "Y", "D", if (design == "X") paste0("X", 1:6))
    panels[[design]][[length(panels[[design]]) + 1]] <- cbind(panel = tag, d[, keep])

    g <- gsynth(Y ~ D, data = d, index = c("id", "time"), force = "time",
                CV = 0, r = GSYNTH_R[[design]], se = TRUE, parallel = FALSE,
                inference = "parametric")
    refs[[length(refs) + 1]] <- data.frame(
      panel = tag, design = design, N = N, TT = TT, T0 = T0, arm = "gsynth",
      niter = NA_integer_, att = g$att.avg, ci_l = g$est.avg[3], ci_u = g$est.avg[4])

    for (ni in c(10000L, 1500L)) {
      b <- pblasso(data = d, index = c("id", "time"), Yname = "Y", Dname = "D",
                   Xname = NULL, Zname = NULL, Aname = NULL, re = "time", r = 10,
                   niter = ni, burn = as.integer(ni / 5), xlasso = 0, zlasso = 0,
                   alasso = 0, flasso = 1)
      e <- effSummary(b, usr.id = NULL, burn = 0, cumu = FALSE, rela.period = TRUE)
      refs[[length(refs) + 1]] <- data.frame(
        panel = tag, design = design, N = N, TT = TT, T0 = T0, arm = "pblasso",
        niter = ni, att = e$est.avg[1], ci_l = e$est.avg[2], ci_u = e$est.avg[3])
    }
    cat(sprintf("seam %s done\n", tag))
  }
}
if (wants("seam")) {
  for (design in names(panels)) {
    write.csv(do.call(rbind, panels[[design]]),
              file.path(out.dir, sprintf("seam_panels_%s.csv", design)), row.names = FALSE)
  }
  write.csv(do.call(rbind, refs), file.path(out.dir, "seam_reference.csv"), row.names = FALSE)
}

## ------------------------------------------------------------------ ##
## 3. Provenance.
## ------------------------------------------------------------------ ##

prov <- sprintf(paste0(
  '{\n  "generated_at": "%s",\n  "only": "%s",\n  "seed": %d,\n',
  '  "r_version": "%s",\n  "platform": "%s",\n',
  '  "packages": {"gsynth": "%s", "pblasso": "%s"},\n',
  '  "note": "each CSV carries its own draw count in a draws column",\n',
  '  "source": "Harvard Dataverse doi:10.7910/DVN/B6SWA1 -- code/simulateCalib.R,',
  ' code/summary_function.R, 9_sim_single_r8.R, 10_sim_single_X.R"\n}\n'),
  format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), only, seed,
  R.version.string, R.version$platform,
  as.character(packageVersion("gsynth")), as.character(packageVersion("pblasso")))
writeLines(prov, file.path(out.dir, "provenance.json"))

cat("wrote --only=", only, " output to ", out.dir, "\n", sep = "")
