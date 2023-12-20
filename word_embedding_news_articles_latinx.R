# Script name:          word_embedding_news_articles_latinx.R
# Created on:           December_4_2023
# Author:               Dr. Martin Calvino
#                       This code was modified from the one presented in Chapter 5 of the book 'Supervised Machine Learning For Text Analysis in R' by Emil Hvitfeldt and Julia Silge
# Purpose:              Extract the semantic meaning of the term "latinx" according to 6 different newspapers
# Version:              v1.11.04.2023

# Data source:          news articles mentioning the term "latinx" were manually downloaded from The New York Times - The Washington Post - Miami Herald - Houston Chronicle - Chicago Tribune - Los Angeles Times 
#                       .csv files containing news articles were manually created for each newspaper

# load libraries
library(tidyverse)
library(tidytext)
library(SnowballC)
library(slider)
library(widyr)
library(furrr)
library(textdata)

# load .csv file containing news articles from The New York Times
nyt_news <- read_csv(file.choose())
View(nyt_news)

# filter out words used 3 times or less in the dataset
# create a tidy text dataset (one token per row)
tidy_nyt_news <- nyt_news %>%
  select(News_ID, Text) %>%
  unnest_tokens(word, Text) %>%
  add_count(word) %>%
  filter(n >= 4) %>%
  select(-n)

View(tidy_nyt_news)

# create nested dataframe with one row per news article
nyt_nested_words <- tidy_nyt_news %>%
  nest(words = c(word))

nyt_nested_words
View(nyt_nested_words)

# create a slide_windows() function to implement fast sliding window computations
# calculate skipgram probabilities
# define a fixed-size moving window that centers around each word
slide_windows <- function(tbl, window_size) {
  skipgrams <- slider::slide(
    tbl,
    ~.x,
    .after = window_size - 1,
    .step = 1,
    .complete = TRUE
  )
  safe_mutate <- safely(mutate)
  
  out <- map2(skipgrams, 1:length(skipgrams), ~ safe_mutate(.x, window_id = .y))
  
  out %>%
    transpose() %>%
    pluck("result") %>%
    compact() %>%
    bind_rows()
}

plan(multisession) # for parallel computing

# calculate PMI (Pointwise Mutual Information) values for each sliding window with size of 4 words 
nyt_tidy_pmi <- nyt_nested_words %>%
  mutate(words = future_map(words, slide_windows, 4L)) %>%
  unnest(words) %>%
  unite(window_id, News_ID, window_id) %>%
  pairwise_pmi(word, window_id)
# when PMI is high, the two words are associated with each other (likely to occur together)
nyt_tidy_pmi
View(nyt_tidy_pmi)
summary(nyt_tidy_pmi[,3]) # the median PMI value is 1.42

# determine word vectors from PMI values using SVD (Singular Value Decomposition: a method for dimensionality reduction via matrix factorization)
# project the sparse, high-dimensional set of word features into a more dense, 100-dimensional set of features
# each word is represented as a numeric vector in this new feature space
nyt_tidy_word_vectors <- nyt_tidy_pmi %>%
  widely_svd(item1, item2, pmi, nv = 100, maxit = 1000)

View(nyt_tidy_word_vectors)

# explore nyt word embedding
# which words are close to each other in this 100-dimensional feature space?
# create a function to find nearest words based on cosine similarity
# return a dataframe sorted by similarity to my word/token of interest
nearest_neighbors <- function(df, token) {
  df %>%
    widely(
      ~ {
        y <- .[rep(token, nrow(.)), ]
        res <- rowSums(. * y) / (sqrt(rowSums(. ^ 2)) * sqrt(sum(.[token, ] ^ 2)))
        matrix(res, ncol = 1, dimnames = list(x = names(res)))
        },
      sort = TRUE
    )(item1, dimension, value) %>%
    select(-item2)
}

# what words are closest to "latinx"
nyt_tidy_word_vectors %>%
  nearest_neighbors("latinx")

################################################################################

# create a function to contain all the above code 
# and apply it to find the closest words to "latinx" in news articles
# from the other 5 newspapers

closest_words <- function(newspaper_news, token) {
  tidy_news <- newspaper_news %>%
    select(News_ID, Text) %>%
    unnest_tokens(word, Text) %>%
    add_count(word) %>%
    filter(n >= 4) %>%
    select(-n)
  
  nested_words <- tidy_news %>%
    nest(words = c(word))
  
  slide_windows <- function(tbl, window_size) {
    skipgrams <- slider::slide(
      tbl,
      ~.x,
      .after = window_size - 1,
      .step = 1,
      .complete = TRUE
    )
    safe_mutate <- safely(mutate)
    
    out <- map2(skipgrams, 1:length(skipgrams), ~ safe_mutate(.x, window_id = .y))
    
    out %>%
      transpose() %>%
      pluck("result") %>%
      compact() %>%
      bind_rows()
  }
  
  plan(multisession)
  
  tidy_pmi <- nested_words %>%
    mutate(words = future_map(words, slide_windows, 4L)) %>%
    unnest(words) %>%
    unite(window_id, News_ID, window_id) %>%
    pairwise_pmi(word, window_id)
  
  tidy_word_vectors <- tidy_pmi %>%
    widely_svd(item1, item2, pmi, nv = 100, maxit = 1000)
  
  nearest_neighbors <- function(df, token) {
    df %>%
      widely(
        ~ {
          y <- .[rep(token, nrow(.)), ]
          res <- rowSums(. * y) / (sqrt(rowSums(. ^ 2)) * sqrt(sum(.[token, ] ^ 2)))
          matrix(res, ncol = 1, dimnames = list(x = names(res)))
        },
        sort = TRUE
      )(item1, dimension, value) %>%
      select(-item2)
  }
  
  tidy_word_vectors %>%
    nearest_neighbors("latinx")
}

################################################################################

# Los Angeles Times
# load .csv file containing news articles from Los Angeles Times
lat_news <- read_csv(file.choose())
View(lat_news)
# find closest words to "latinx" in the word embedding from news articles 
closest_words(lat_news, "latinx")


# Houston Chronicle
# load .csv file containing news articles from Houston Chronicle
huc_news <- read_csv(file.choose())
View(huc_news)
# find closest words to "latinx" in the word embedding from news articles 
closest_words(huc_news, "latinx")


# Miami Herald
# load .csv file containing news articles from Miami Herald
mihe_news <- read_csv(file.choose())
View(mihe_news)
# find closest words to "latinx" in the word embedding from news articles 
closest_words(mihe_news, "latinx")


# The Washington Post
# load .csv file containing news articles from Washington Post
wp_news <- read_csv(file.choose())
View(wp_news)
# find closest words to "latinx" in the word embedding from news articles 
closest_words(wp_news, "latinx")


# Chicago Tribune
# load .csv file containing news articles from Chicago Tribune
ct_news <- read_csv(file.choose())
View(ct_news)
# find closest words to "latinx" in the word embedding from news articles 
closest_words(ct_news, "latinx")
