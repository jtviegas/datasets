# datasets

[![cicd](https://github.com/jtviegas/datasets/actions/workflows/cicd.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/cicd.yml)
![Coverage](./coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/tgedr-datasets)](https://pypi.org/project/tgedr-datasets/)

## datalake pipelines


[![tickers](https://github.com/jtviegas/datasets/actions/workflows/tickers.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/tickers.yml)
[![prices](https://github.com/jtviegas/datasets/actions/workflows/prices.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/prices.yml) 
[![articles](https://github.com/jtviegas/datasets/actions/workflows/articles.yml/badge.svg)](https://github.com/jtviegas/datasets/actions/workflows/articles.yml)


### articles

![articles distribution per ticker](assets/articles_distribution.png)

![number of articles found per run](assets/articles_per_time.png)

[download @ hugging face](https://huggingface.co/datasets/jtviegas/ticker_analysis_articles)

### prices

![prices distribution per ticker](assets/prices_distribution.png)

![number of prices found per run](assets/prices_per_time.png)

[download @ hugging face](https://huggingface.co/datasets/jtviegas/ticker_analysis_prices)

### tickers

![tickers distribution](assets/tickers_distribution.png)

![tickers of articles found per run](assets/tickers_per_time.png)

[download @ hugging face](https://huggingface.co/datasets/jtviegas/ticker_analysis_tickers)

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