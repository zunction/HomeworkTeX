data.long %>% 
  ggplot(aes(date, prcp)) + 
    geom_point() + 
    facet_wrap(~month)

to2chars <- function(x) {
  sapply(x, function(y) ifelse(y < 10, 
                               paste0("0", y),
                               as.character(y)))
}

# bigplot <- data.long %>% 
#   mutate(station = to2chars(station),
#          month = to2chars(month),
#          group = paste0(station,'-',month)) %>% 
#   ggplot(aes(date, prcp)) + 
#     geom_point() + 
#     facet_wrap(~group, ncol = 24)

pdf('Plots/bigplot.pdf', width = 11.69, height = 16.53)
print(bigplot)
dev.off()

data.long %>% 
  ggplot(aes(date, prcp)) + 
  geom_point() + 
  facet_wrap(~station) +
  facet_wrap(~month)

plotMonth <- function(m) {
  data.long %>% 
    filter(month == m) %>% 
  ggplot(aes(date, prcp)) + 
    geom_point() + 
    facet_wrap(~station) +
    ggtitle(month.abb[m]) %>% 
    print()
}

lapply(1:12, plotMonth)
