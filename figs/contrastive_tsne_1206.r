library(tidyverse)
library(ggplot2)
setwd("~/Dropbox/Documents/tri-i/FoDS/project/analysis")

exp = read_csv("221206_contrastive_5v_52_no_activation_tsne.csv") %>%
  separate(experiment, into = c("celltype", "exp_n"), remove = F)

ggplot(exp) +
  geom_point(aes(x = tsne_x, y = tsne_y, color = parent)) +
  theme_bw() +
  guides(color=guide_legend(ncol=1))
ggsave('1206_noact_contrastive_tsne_pclass.pdf', device = 'pdf', height = 8, width = 10)

ggplot(exp) +
  geom_point(aes(x = tsne_x, y = tsne_y, color = experiment)) +
  theme_bw()
ggsave('1206_noact_contrastive_tsne_exp.pdf', device = 'pdf', height = 8, width = 10)

exp = read_csv("221206_contrastive_5v_52_tsne.csv") %>%
  separate(experiment, into = c("celltype", "exp_n"), remove = F)

ggplot(exp) +
  geom_point(aes(x = tsne_x, y = tsne_y, color = parent)) +
  theme_bw() +
  guides(color=guide_legend(ncol=1))
ggsave('1206_contrastive_tsne_pclass.pdf', device = 'pdf', height = 8, width = 10)

ggplot(exp) +
  geom_point(aes(x = tsne_x, y = tsne_y, color = experiment)) +
  theme_bw()
ggsave('1206_contrastive_tsne_exp.pdf', device = 'pdf', height = 8, width = 10)
