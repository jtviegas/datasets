# datasets

[![cicd](https://github.com/jtviegas/datasets/actions/workflows/cicd.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/cicd.yml)
![Coverage](./coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/tgedr-dataset)](https://pypi.org/project/tgedr-dataset/)

## datalake pipelines


[![tickers](https://github.com/jtviegas/datasets/actions/workflows/tickers.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/tickers.yml)
[![prices](https://github.com/jtviegas/datasets/actions/workflows/prices.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/prices.yml) 
[![articles](https://github.com/jtviegas/datasets/actions/workflows/articles.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/articles.yml)


### articles

![articles distribution per ticker](assets/article_distribution.png)

![number of articles found per run](assets/articles_per_processing_time.png)

### download datasets

[ticker_analysis @ huggingface](https://huggingface.co/datasets/jtviegas/ticker_analysis)

## development
- main requirements:
  - _uv_  
  - _bash_
- Clone the repository like this:

  ``` bash
  git clone git@github.com:jtviegas/datasets
  ```
- cd into the folder: `cd datasets`
- install requirements: `./helper.sh reqs`