### CIS 4130
### Isaias Urena
### Isaias.Urena@baruchmail.cuny.edu
##

### Project Proposal
The aim of this project is to identify the characteristics related to a specific type of visual art. Through the collection and evaluation of images from WikiArt.org, the likelihood of a new image falling into an art category can be predicted. Unique data can be generated following the same attributes observed in each art type. 

Dataset - https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan 

This dataset is a collection of 96014 visual art pieces, which include abstract, portrait, and landscape paintings. Each piece varies color composition, objects depicted, and the year they were created. I plan to evaluate all the images in each category to determine any similarities, traits, and differences they have from one another. These traits could be color composition or shapes and will be used to determine the likelihood a new image falls under the category.
 
### Data Collection

To begin, I create a bucket in the Amazon s3 under the name “project-data-images” to store the dataset by using, 
`aws s3api create-bucket --bucket project-data-images --region us-east-2 --create-bucket-configuration LocationConstraint=us-east-2`.
Since the data comes from Kaggle, I’ll install the Kaggle CLI. I’ll be modifying the directory and pasting in my username and key from my API Token (kaggle.json). This was acquired from my Kaggle account page under “New API Token”. After, I call for a list of dataset to ensure CLI is working appropriately.  
`mkdir .kaggle`

`nano .kaggle/kaggle.json`

  *copied and pasted information from Kaggle.json then save and exit*

`chmod 600 .kaggle/kaggle.json`

`kaggle datasets list`

Following this, I access the Kaggle_api_extended.py and modify lines 1582 and 1594 by expanding on the conditionals.

`nano  ~/.local/lib/python3.7/site-packages/kaggle/api/kaggle_api_extended.py`

Line 1582 changed to:
```diff
- if not os.path.exists(outpath):

+ if not os.path.exists(outpath) and outpath != "-":
```
Line 1594 changed to:
```diff
- with open(outfile, 'wb') as out:

+ with open(outfile, 'wb') if outpath != "-" else os.fdopen(sys.stdout.fileno(), 'wb', closefd=False) as out:
```
Once completed, I’m able to download the dataset. The dataset in question can be located through the username/title in that order. Ipythonx/wikiart-gangogh-creating-art-gan is the dataset and was found within the URL link on the Kaggle website. I’ll be piping the data into the aws s3 bucket named “project-data-images” and into a zip file named “art.zip”.

`kaggle datasets download -d ipythonx/wikiart-gangogh-creating-art-gan -p -  | aws s3 cp - s3://project-data-images/art.zip`

After the download was completed, I checked to see if the bucket is storing any information with the following command
`aws s3 ls s3://project-data-images/`.

![image](https://user-images.githubusercontent.com/101361036/208178021-4776dd48-95b6-4c9f-a2a5-21349e4d7d29.png)
 
Following this, I run a python3 script that unzips the art.zip file and ensures that the content is accessible. The script consists of:
```py
import zipfile
import boto3
from io import BytesIO
bucket="project-data-images" 
zipfile_to_unzip="art.zip" 
s3_client = boto3.client('s3', use_ssl=False)
s3_resource = boto3.resource('s3')
# Create a zip object that represents the zip file
zip_obj = s3_resource.Object(bucket_name=bucket, key=zipfile_to_unzip)
buffer = BytesIO(zip_obj.get()["Body"].read())
z = zipfile.ZipFile(buffer)
# Loop through all of the files contained in the Zip archive
for filename in z.namelist():
	print('Working on ' + filename)
	# Unzip the file and write it back to S3 in the same bucket
	s3_resource.meta.client.upload_fileobj(z.open(filename), Bucket=bucket, Key=f'{filename}')
```
To ensure the data was properly downloaded, I navigate the project-data-images bucket using the aws s3 ls command to confirm the individual folders were download. 
`aws s3 ls s3://project-data-images/images/`
	(images/ being the newly added folder)
  
![image](https://user-images.githubusercontent.com/101361036/208178225-80f24d45-9135-4a9d-b85a-06370f6d7825.png)

 
### Analyzing Data
With the creation of an accessible folder, the included .csv file will be used to count how many images are in each file. The WikiArt.csv file includes three columns, which are image_path, class_name, and label. By grouping all entry by there class_name, we can then count how many times each label is brought up. Each label corresponds to a class_name. Using the following script, we can display each class_name with their respective count. The results will then be written down in a .csv file named ‘WikiArt_analysis.csv’
```py
import boto3
import pandas as pd
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('project-data-images')
df = pd.read_csv('s3://project-data-images/WikiArt.csv')
results = df.groupby('class_name').labels.agg(['count'])
results.to_csv('s3://project-data-images/WikiArt_analysis.csv')
print(results)
```

![image](https://user-images.githubusercontent.com/101361036/208178268-03046c0a-950a-42a4-beff-e8f4bb2f1a86.png)

The current .csv file doesn’t provide information on the images other than the category they fall under. I’ll be running the script below to create a new .csv file, named “WikiArtFeatures.csv”, with columns describing different features of each image. These features include resolution, RGB pixel mean, light ratio, and the number of key points from running the ORB, FAST, and BRIEF algorithms.
```py
# necessary imports
from skimage.io import imread, imshow
import boto3
import botocore.exceptions
import csv
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# s3 bucket
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('project-data-images')
fields = ['Image_name', 'Class_name', 'HorizontalResolution', 'VerticalResolution',\
	 'LightRatio', 'RGBMean','ORB', 'FAST', 'BRIEF']
rows = []
for object in my_bucket.objects.all():
	if 'jpg' in object.key:
		print(f"Working with: {object.key}")
		name = object.key.split('/')[-1]
		start = object.key.index('/')
		end = object.key.index('/',start+1)
		class_name = object.key[start+1:end]
		try:
			my_bucket.download_file(object.key, name)
			# read image in RGB
			try:
				img = imread(name)
			except:
				print("Couldn't read file")
				continue
			try:
				width, height = img.shape[:2]
			except:
				width, height = img.shape

			pixel = np.sum(width*height)
			rgbmean = np.average(img)
			# Computing ORB
			orb = cv.ORB_create(nfeatures=50000)
			orbkeypoints, orbdescriptor = orb.detectAndCompute(img, None)
			# computing FAST
			fast = cv.FastFeatureDetector_create()
			fastkeypoints = fast.detect(img, None)
			# computing BRIEF
			brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
			briefkeypoints, briefdescriptor = brief.compute(img,fastkeypoints)
			# computing Light ratio
			img = imread(name, as_gray=True)
			lightratio = np.sum(img)/pixel
			# creating entries
entry = [name, class_name, width, height,lightratio,\
rgbmean,len(orbkeypoints), len(fastkeypoints),len(briefkeypoints)]
			rows.append(entry)
		except botocore.exceptions.DataNotFoundError as e:
			print(e)
# create .csv from dataframe
df=pd.DataFrame(data = rows, columns = fields)
df.to_csv('s3://project-data-images/WikiArtFeatures.csv', index=False)
```
With the new .csv file created; I’ll be able to collect information regarding the image’s features. To ensure the new file is accessible I’ll be taking the average, maximum, and minimum of image resolutions. I’ll be reusing the prior script with the addition of our wanted parameters. 
```py
import boto3
import pandas as pd
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('project-data-images')
df = pd.read_csv('s3://project-data-images/WikiArtFeatures.csv')
results = df.groupby('Class_name').agg({'VerticalResolution': ['count','mean','min','max'],	'HorizontalResolution': ['mean','min','max']})
print(results)
```

![image](https://user-images.githubusercontent.com/101361036/208178473-b7288bbb-3695-4dd5-941d-d09cfddbec52.png)

 
### Modeling
After having collected some features into an .csv file, I begin modeling a pipeline using pyspark. The .csv file collected from the prior step will be read, split up, then fitted into the pipeline. To begin, I first read the file into “wikiartdf” and create a new column to store the label for each image. The labels will be a number from zero to thirteen, which corresponds to an art genre.
```py
# Necessary Imports
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *
from itertools import chain
from pyspark.sql.types import *
import numpy as np

spark = SparkSession.builder.getOrCreate()

#Reading csv file
wikiartdf = spark.read.csv("s3://project-data-images/WikiArtFeatures.csv/", header=True, inferSchema=True)
#Creating labels
labels = {'abstract':0, 'animal-painting':1, 'cityscape':2, 'figurative':3, 'flower-painting':4, \
'genre-painting':5, 'landscape':6, 'marina':7, 'mythological-painting':8, 'nude-painting-nu':9, \
'portrait':10, 'religious-painting':11, 'still-life':12, 'symbolic-painting':13}
labels_expr = create_map([lit(x) for x in chain(*labels.items())])
wikiartdf = wikiartdf.withColumn("label", labels_expr[col("Class_name")])
```
Since two of the ten columns are strings, I'll be using an String Indexer and One Hot Encoder to convert them into a vector before assembling all the columns in the Vector Assembler. Having made these stages, the pipeline can be assigned to “wikiartpipe”.
```py
#Indexer, encoder for two strings
indexer = StringIndexer(inputCols=["Image_name", "Class_name"], outputCols=["nameIndex","classIndex"])
encoder = OneHotEncoder(inputCols=["nameIndex", "classIndex"], outputCols = ["nameVector", "classVector"], dropLast=False)
assembler = VectorAssembler(inputCols=["nameVector", "classVector", "HorizontalResolution", "VerticalResolution",\
 "LightRatio", "RGBMean", "ORB", "FAST", "BRIEF", "label"], outputCol="features")
#Pipeline
wikiartpipe = Pipeline(stages=[indexer, encoder, assembler])
```
This pipeline is then fitted with the wikiartdf dataframe and split up into a trainingData and testData. I’ve made the distribution of data to be a 70/30 split. For this pipeline, I’ll be using logistic regression as an estimator and multiclass classification as an evaluator. The Cross Validator is assigned to “cv” before fitting in the training data. 
```py
#Transformed pipeline
transformed_wikiartdf = wikiartpipe.fit(wikiartdf).transform(wikiartdf)
#Spliting data
trainingData, testData = wikiartdf.randomSplit([0.7,0.3])
#Logistic Regression
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
#Creating grid
grid = ParamGridBuilder()
grid = grid.addGrid(lr.regParam, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
grid = grid.addGrid(lr.elasticNetParam, [0, 1])
grid = grid.build()
#Multiclass classificaiton evaluator
evaluator = MulticlassClassificationEvaluator()
#Cross Validator
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
all_models = cv.fit(trainingData)
```
I extract the best model derived from the Cross Validator and test the accuracy of its predictions. 
```py
#Best model
bestModel = all_models.bestModel
#Results from best model
test_results = bestModel.transform(testData)
#Show each image's prediction
test_results.select('Image_name','Class_name','probability', 'prediction',
'label').show(truncate=False)
#Best models accuaracy 
print(evaluator.evaluate(test_results))
```
The model’s accuracy, precision, recall, and F1 score are all calculated.
```py
def calculate_recall_precision(cm):
	tn = cm[0][1] # True Negative
	fp = cm[0][2] # False Positive
	fn = cm[1][1] # False Negative
	tp = cm[1][2] # True Positive
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	f1_score = 2 * ((precision * recall) / (precision + recall))
	return accuracy, precision, recall, f1_score

cm = test_results.groupby('label').pivot('prediction').count().fillna(0).collect()
print(calculate_recall_precision(cm))
```
This model is then saved into the “project-data-images” bucket under the name “wikiart_logistic_regression”.
```
#Saving the best model
model_path = "s3://project-data-images/wikiart_logistic_regression"
bestModel.write().overwrite().save(model_path) 
```
### Visualization
Visualizing the results of any model makes it easier to understand how successful a model has become without having to be overwhelmed by multiple values. The results from the prior section, “test_results”, can be used to create a confusion matrix. This will display each art genre with their predictions. 
```py
#Confusion Matrix based on results from best model
test_results.groupby('label').pivot('prediction').count().fillna(0).show()
```

![image](https://user-images.githubusercontent.com/101361036/208178523-f330c91b-e95a-48a1-b320-1ff5f38f9c48.png)

We can create a graph to show the relationship between the model's precision and recall.
```py
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot(bestModel.summary.precisionByLabel, bestModel.summary.recallByLabel)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title("Precision and Recall Curve")
plt.savefig("s3://project-data-images/PRC.png")
```
The created .png can then be moved from the HDFS by using
`hdfs df -get hdfs:///PRC.png` then `aws s3 cp PRC.png s3://project-data-images`

PRC.png:




