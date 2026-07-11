library(splines)
library(fda)
library(ggplot2)

data = Data_Germany
X = data[,2:32]
x = seq(0,1,length=31)

splinebasis = create.bspline.basis(c(0,1),10)
smooth = smooth.basis(x,t(X),splinebasis)
Xfun = smooth$fd
pca = pca.fd(Xfun, 10)
var.pca = cumsum(pca$varprop)
nharm = sum(var.pca < 0.95) + 1
pc = pca.fd(Xfun, nharm)

cluster_x = as.matrix(pc$scores)
cluster_x <- scale(cluster_x)
# Determine number of clusters
wss <- (nrow(cluster_x)-1)*sum(apply(cluster_x,2,var))
for (i in 2:8) wss[i] <- sum(kmeans(cluster_x,
                                     centers=i)$withinss)
plot(1:8, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

# K-Means Cluster Analysis
fit <- kmeans(cluster_x, 3) # 5 cluster solution

# append cluster assignment
data <- data.frame(data, fit$cluster)
data[,c(1,46)]
western_germany_cluster = fit$cluster[7]
new_data = data[data[,46]==western_germany_cluster,1:45]
#write.csv(new_data,"clustered_data.csv")

########## Good Curves
x = 1:10
data_1 = as.data.frame(cbind(x,var.pca))
ggplot(data_1,aes(x=x, y=var.pca)) +
  geom_line(size=1)+scale_x_continuous(breaks=1:10)+
  geom_point()+xlab("Number of FPC-scores") + ylab("Proportion of Explained Variation")+theme_bw()

x=1:8
data_2 = as.data.frame(cbind(x,wss))
ggplot(data_2,aes(x=x, y=wss)) +
  geom_line(size=1)+scale_x_continuous(breaks=1:8)+
  geom_point()+xlab("Number of Clusters") + ylab("Within Groups Sum of Squares")+theme_bw()

########### Silhoute
library(cluster)

k.max = 8
sil = rep(0, k.max)

# Compute the average silhouette width for 

for(i in 2:k.max){
  tmp = kmeans(cluster_x, centers = i, nstart = 10)
  ss <- silhouette(tmp$cluster, dist(cluster_x))
  sil[i] <- mean(ss[, 3])
}

# Plot the  average silhouette width
plot(1:k.max, sil, type = "b", pch = 19, 
     frame = FALSE, xlab = "Number of clusters k")
abline(v = which.max(sil), lty = 2)

x=1:k.max
data_3 = as.data.frame(cbind(x,sil))
ggplot(data_3,aes(x=x, y=sil)) +
  geom_line(size=1)+scale_x_continuous(breaks=1:k.max)+
  geom_point()+xlab("Number of Clusters") + ylab("Silhouette Coefficient")+theme_bw()

