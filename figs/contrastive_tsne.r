library(tidyverse)
library(ggplot2)
setwd("~/Dropbox/Documents/tri-i/FoDS/project/analysis")

exp = read_csv("contrastive_tsne_exp.csv") %>%
  separate(experiment, into = c("celltype", "exp_n"), remove = F)

ggplot(exp) +
  geom_point(aes(x = tsne_x, y = tsne_y, color = parent)) +
  theme_bw() +
  guides(color=guide_legend(ncol=1))
ggsave('contrastive_tsne_pclass.pdf', device = 'pdf', height = 8, width = 14)

ggplot(exp) +
  geom_point(aes(x = tsne_x, y = tsne_y, color = experiment)) +
  theme_bw()
ggsave('contrastive_tsne_exp.pdf', device = 'pdf', height = 8, width = 14)
