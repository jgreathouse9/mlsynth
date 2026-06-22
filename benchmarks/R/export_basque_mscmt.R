suppressMessages({library(Synth); library(MSCMT)})
data(basque)
Basque <- listFromLong(basque, unit.variable="regionno", time.variable="year",
                       unit.names.variable="regionname")
school.sum <- with(Basque, colSums(school.illit+school.prim+school.med+school.high+school.post.high))
Basque$school.higher <- Basque$school.high + Basque$school.post.high
for (item in c("school.illit","school.prim","school.med","school.higher"))
  Basque[[item]] <- 5*100*t(t(Basque[[item]])/school.sum)
vars <- c("gdpcap","sec.agriculture","sec.energy","sec.industry","sec.construction",
          "sec.services.venta","sec.services.nonventa","school.illit","school.prim",
          "school.med","school.higher","invest","popdens")
long <- NULL
for (v in vars){
  m <- Basque[[v]]                       # rows=years, cols=units
  df <- data.frame(year=as.integer(rownames(m)),
                   stack(as.data.frame(m, check.names=FALSE)))
  names(df) <- c("year","value","regionname")
  df$variable <- v
  long <- rbind(long, df)
}
wide <- reshape(long, idvar=c("year","regionname"), timevar="variable", direction="wide")
names(wide) <- sub("^value\\.", "", names(wide))
write.csv(wide, "/tmp/basque_mscmt.csv", row.names=FALSE)
cat("rows", nrow(wide), "cols", paste(names(wide), collapse=","), "\n")
# sanity: schooling non-NA years
cat("school.illit non-NA years:", paste(sort(unique(wide$year[!is.na(wide$school.illit)])), collapse=","), "\n")
