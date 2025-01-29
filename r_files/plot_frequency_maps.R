## Simple install and load function for CRAN packages
load_install <- function(pkg_name) {
  if(!pkg_name %in% rownames(installed.packages())) {
    install.packages(pkg_name, repos = "http://cran.us.r-project.org")
  }
  library(pkg_name, character.only = TRUE)
}

# install packages
load_install('here')
load_install('sf')
load_install('tidyverse')
load_install('RColorBrewer')
load_install('fs')
load_install('remotes')
load_install("ini")

path <- file.path(here() %>% dirname())
print(path)

tmap_flag <- 0
if('tmap' %in% rownames(installed.packages())) tmap_flag <- ifelse(as.numeric(substr(packageVersion("tmap"), 1,3)) < 3.9, 1, 0)

if(!'tmap' %in% rownames(installed.packages()) | tmap_flag == 1){
   # If package was not able to be loaded then re-install
  remotes::install_github('r-tmap/tmap', dependencies=T, upgrade='never')
   # Load package after installing
  require('tmap', character.only = TRUE )
}
# otherwise not loaded
library(tmap)

is.integer0 <- function(x){
  is.integer(x) && length(x) == 0L
}

# get files
config <- read.ini(paste0(path,'/input/config.ini'))
hru <- read_sf(paste0(path, '/input/',config$Data$input_shapefile))
csv <- dir(paste0(path,'/output'), pattern='clusters_representativesolutions', full.names=TRUE)
dat <- read.csv(csv[which.max(file_info(csv)$change_time)]) %>% 
  mutate(id = 1:nrow(.))


dat_long <- dat %>% 
  select(id, starts_with('UNIT')) %>% 
  pivot_longer(., cols = -id) %>% filter(!is.na(value)) %>% 
  mutate(lu = factor(value, levels = 1:max(value), labels = paste0('land use option ', c(1:max(value)))),
         name = str_remove(name, 'UNIT_'))

# number of clusters and land use options
n_clust <- max(as.numeric(as.character(dat$Cluster)), na.rm = TRUE) #changed to exclude "outlier"
n_lu <- length(unique(dat_long$lu))

# names of land use options
names_lu <- config$Frequency_Plots$names_lu
names_lu <- unlist(strsplit(names_lu,", "))

if(is.null(names_lu)) names_lu <- paste0('Option ', c(1:n_lu))

if(file.exists(paste0(path, '/output/freq_map_cluster_0.png'))){
  #file.remove(dir(paste0(path, '/output/'), pattern='freq_map_cluster_', full.names = T))
  files_to_delete <- dir(paste0(path, '/output/'), pattern='^freq_map_cluster_[0-9]+\\.png$', full.names = TRUE)
  file.remove(files_to_delete)
}

for(i in 0:n_clust){
  plt_dat <- dat
  sel_id <- which(plt_dat$Cluster == i)
  
  is_filtered <- length(unique(plt_dat$id)) != length(unique(dat_long$id))
  if(!is.null(sel_id) | is_filtered){
    plt_dat$filter_id <- 1:nrow(plt_dat)
    if (!is.null(sel_id)) {
      plt_dat <- filter(plt_dat, filter_id %in% (sel_id + 1))
    }
    dat_sel <- filter(dat_long, id %in% plt_dat$id)
  } else {
    dat_sel <- dat_long
  }
  
  dat_sel <- dat_sel %>%
    group_by(name, lu) %>%
    summarise(fract = n(), .groups = 'drop_last') %>%
    mutate(fract = round(fract / sum(fract) * 100, digits = 0)) %>%
    pivot_wider(., names_from = lu, values_from = fract)
  
  dat_sel2 <- dat_sel[,-1]
  #here is the problem, lu_dominant is giving the columns not the lu option, my data might be ordered differently though
  lu_dominant <- apply(dat_sel2, 1, which.max)
  lu_dominant = as.numeric(sub("^land use option", "", colnames(dat_sel2)[lu_dominant]))
  
  dat_sel <- cbind.data.frame(dat_sel, lu_dominant)
  
  hru_sel <- left_join(hru, dat_sel, by = c('UNIT' = 'name'))
  
  # plotting
  if(paste0(path, '/output/freq_map_cluster_', i, '.png') %in% dir(paste0(path, '/output/'), full.names = T)){
    file.remove(paste0(path, '/output/freq_map_cluster_', i, '.png'))
  }
  
  tmap_options(frame = F, # Plot panels are plotted without frames
               asp = 1.5, # Aspect ratio of plot panels. Change if place for legend is too narrow. value > 1 = plot wider
               #max.categories = Inf,
               component.autoscale = T)
  
  # define the intervals of the scale bar, depending on the catchment size.
  scale_bar_intervals <- c(0, 2.5, 5,10,15,30) #km
  
  pal <- c("Reds", "Blues", "Greens", "Greys", "Purples")
  
  map_frame <- tm_shape(hru) + #adjust name of map
    tm_borders(col='grey20') +
    tm_title(paste0('Cluster ', i, ':\nDominant land use and frequency [%]'),
             size = 1.2,
             position = tm_pos_out(cell.v = 'top', 
                                   cell.h = 'center',
                                   pos.h = 'center',
                                   pos.v = 'bottom'))
  
  if(n_lu==1){
    freq_map <- map_frame +
      
      tm_shape(hru_sel[which(hru_sel$lu_dominant==1 | is.na(hru_sel$lu_dominant)),]) +
      tm_fill(fill = "land use option 1",
              fill.scale = tm_scale(values = 'brewer.reds',
                                    n=5,
                                    label.na = "not considered",
                                    value.na = 'lightgoldenrodyellow'),
              fill.legend = tm_legend(names_lu[1],
                                      frame = F,
                                      position = tm_pos_out(cell.h = 'right', 
                                                            cell.v = 'center',
                                                            pos.v = 'center'),
                                      text.size = 0.7,
                                      title.size = 1))
    
    tmap_save(freq_map, 
              paste0(path, '/output/freq_map_cluster_', i, '.png'),
              width = 7,
              height = 7)
      
  }
  
  if(n_lu==2){
    
    if(any(!is.na(hru_sel$`land use option 1`))){
      map1 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==1 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 1",
                fill.scale = tm_scale(values = 'brewer.reds',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[1],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'top'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map1 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 2`))){
      map2 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==2 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 2",
                fill.scale = tm_scale(values = 'brewer.blues',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[2],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'bottom'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map2 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    freq_map <- map_frame + map1 + map2
    
    tmap_save(freq_map, 
              paste0(path, '/output/freq_map_cluster_', i, '.png'),
              width = 7,
              height = 7)
    
  }
  
  if(n_lu==3){
    
    if(any(!is.na(hru_sel$`land use option 1`))){
      map1 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==1 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 1",
                fill.scale = tm_scale(values = 'brewer.reds',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[1],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'top'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map1 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 2`))){
      map2 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==2 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 2",
                fill.scale = tm_scale(values = 'brewer.blues',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[2],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'center'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map2 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 3`))){
      map3 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==3 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 3",
                fill.scale = tm_scale(values = 'brewer.greens',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[3],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'bottom'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map3 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    freq_map <- map_frame + map1 + map2 + map3
    
    tmap_save(freq_map, 
              paste0(path, '/output/freq_map_cluster_', i, '.png'),
              width = 8,
              height = 8)
    
  }  
  
  if(n_lu==4){
    
    if(any(!is.na(hru_sel$`land use option 1`))){
      map1 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==1 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 1",
                fill.scale = tm_scale(values = 'brewer.reds',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[1],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'top',
                                                              pos.v = 'bottom'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map1 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 2`))){
      map2 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==2 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 2",
                fill.scale = tm_scale(values = 'brewer.blues',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[2],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'top'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map2 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 3`))){
      map3 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==3 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 3",
                fill.scale = tm_scale(values = 'brewer.greens',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[3],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'bottom'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map3 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 4`))){
      map4 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==4 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 4",
                fill.scale = tm_scale(values = 'brewer.greys',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[4],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'bottom',
                                                              pos.v = 'top'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map4 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    freq_map <- map_frame + map1 + map2 + map3 + map4
    
    tmap_save(freq_map, 
              paste0(path, '/output/freq_map_cluster_', i, '.png'),
              width = 7,
              height = 7)
    
  }  
  
  if(n_lu==5){
    
    if(any(!is.na(hru_sel$`land use option 1`))){
      map1 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==1 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 1",
                fill.scale = tm_scale(values = 'brewer.reds',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[1],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'top',
                                                              pos.v = 'bottom'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map1 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 2`))){
      map2 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==2 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 2",
                fill.scale = tm_scale(values = 'brewer.blues',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[2],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'top'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map2 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 3`))){
      map3 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==3 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 3",
                fill.scale = tm_scale(values = 'brewer.greens',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[3],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'center'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map3 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 4`))){
      map4 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==4 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 4",
                fill.scale = tm_scale(values = 'brewer.greys',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[4],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'center',
                                                              pos.v = 'bottom'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map4 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    if(any(!is.na(hru_sel$`land use option 5`))){
      map5 <- tm_shape(hru_sel[which(hru_sel$lu_dominant==5 | is.na(hru_sel$lu_dominant)),]) +
        tm_fill(fill = "land use option 5",
                fill.scale = tm_scale(values = 'brewer.purples',
                                      n=3,
                                      label.na = "not considered",
                                      value.na = 'lightgoldenrodyellow'),
                fill.legend = tm_legend(names_lu[5],
                                        frame = F,
                                        position = tm_pos_out(cell.h = 'right', 
                                                              cell.v = 'bottom',
                                                              pos.v = 'top'),
                                        text.size = 0.7,
                                        title.size = 1))
    }else{
      map5 <- tm_shape(hru) + #adjust name of map
        tm_borders(col=NULL)
    }
    
    freq_map <- map_frame + map1 + map2 + map3 + map4 + map5 
    
    tmap_save(freq_map, 
              paste0(path, '/output/freq_map_cluster_', i, '.png'),
              width = 8,
              height = 8)
    
  } 
  
  rm(sel_id, plt_dat, dat_sel, dat_sel2, lu_dominant, hru_sel, tmap_options)
  
}


