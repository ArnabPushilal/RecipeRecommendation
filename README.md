# RecipeRecommendation

Download the data, pre-trained models and the results from this link and place them in the same folder as the code.

https://drive.google.com/drive/folders/1X61En59RkomZiE3mSvzvmKrUdXLmz_ra?usp=sharing, 
https://drive.google.com/drive/folders/1eTGXWaA_mzijugK6f3GYkMQ52LsLL64F?usp=sharing, 
https://drive.google.com/drive/folders/1qCbKiIDnJ7m1JwnkYhzOKWYcIULVh-vP?usp=sharing

AUC metrics for all models available in this file
https://drive.google.com/file/d/1no2rTPtGhXLmMNPfSwitKUB__FAKBpDf/view?usp=sharing

Other files like the embeddings, similarity dictionary are present in this folder:
https://drive.google.com/drive/folders/1-M5xKI-3sL0a6SBmRwT_Lo08BoJVdTOT?usp=sharing

## A brief description of the python files:

extracting.py --> Contains the code to scrape ingredients

recommender.ipynb --> Contains ad-hoc type code which produces figures, recommends healthy recipes, generate embeddings, EDA etc



models.py --> Contains all the pytorch models

train.py --> Scripts for training each model

run_2.py --> Used to run and test each model (Models commented out in __main__, to run please uncomment the one to be run)

dataloader.py --> Contains pytorch dataloaders for the models

test_model.py --> Script for testing the models.


