# datasets

[![cicd](https://github.com/jtviegas/datasets/actions/workflows/cicd.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/cicd.yml)
![Coverage](./coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/tgedr-datasets)](https://pypi.org/project/tgedr-datasets/)

## datalake pipelines


[![tickers](https://github.com/jtviegas/datasets/actions/workflows/tickers.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/tickers.yml)
[![prices](https://github.com/jtviegas/datasets/actions/workflows/prices.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/prices.yml) 
[![articles](https://github.com/jtviegas/datasets/actions/workflows/articles.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/articles.yml)


### articles

[download @ hugging face](https://huggingface.co/datasets/jtviegas/ticker_analysis_articles)

![articles metrics](metrics/articles_metrics.png)

### prices

[download @ hugging face](https://huggingface.co/datasets/jtviegas/ticker_analysis_prices)

![prices metrics](metrics/prices_metrics.png)

### tickers

[download @ hugging face](https://huggingface.co/datasets/jtviegas/ticker_analysis_tickers)

![tickers metrics](metrics/tickers_metrics.png)

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