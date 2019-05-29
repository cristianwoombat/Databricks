# Databricks notebook source
Correos_df = spark.sql("select cast(sender as string), bodyPreview, receivedDateTime, hasAttachments, importance, cast(subject as string), case when upper(bodyPreview) like '% UIS %' AND cast(sender as string) like '%uis%' OR cast(subject as string) like '%cursos%' OR upper(bodyPreview) like '%CORDIAL%' OR importance like '%high%' or hasAttachments then 1 else 0 end Fraude from correos700_json ")
Correos_df.createOrReplaceTempView("correos")
Correos_df.printSchema()
Correos_df.show(10)

# COMMAND ----------

Correos_df = spark.sql("""select hasAttachments, importance, case when upper(bodyPreview) like '% UIS %' then 1 when upper(bodyPreview) like '% CORDIAL %' then 2 when upper(bodyPreview) like '% CORDIAL %' and upper(bodyPreview) like '% UIS %' then 3 else 0 end Body_UIS, case when sender like '%uis%' then 1 else 0 end Sender_UIS, case when subject like '%cursos%' then 1 when subject like '% uis %' then 2 else 0 end subject_U, Fraude from correos""")

# COMMAND ----------

(train, test) = Correos_df.randomSplit([0.8, 0.2], seed= 12345)
train.cache()
test.cache()

# COMMAND ----------

# Reset the DataFrames for no fraud (`dfn`) and fraud (`dfy`)
dfn = train.filter(train.Fraude == 0)
dfy = train.filter(train.Fraude == 1)

N = train.count()
y = dfy.count()
p = y/N

train_b = dfn.sample(False, p, seed = 92285).union(dfy)

display(train_b.groupBy("Fraude").count())

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

indexer = StringIndexer(inputCol = "importance", outputCol = "importanceIndexed")



va = VectorAssembler(inputCols = ["hasAttachments","importanceIndexed","Body_UIS","Sender_UIS","subject_U"], outputCol = "features")
dt = DecisionTreeClassifier(labelCol = "Fraude", featuresCol = "features", seed = 54321, maxDepth = 5,maxBins = 52)

# COMMAND ----------

pipeline_1 = Pipeline(stages=[ indexer, va, dt])

Model_b = pipeline_1.fit(train_b)
display(Model_b.stages[-1])

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Use BinaryClassificationEvaluator to evaluate our model
evaluatorPR = BinaryClassificationEvaluator(labelCol = "Fraude", rawPredictionCol = "prediction", metricName = "areaUnderPR")
evaluatorAUC = BinaryClassificationEvaluator(labelCol = "Fraude", rawPredictionCol = "prediction", metricName = "areaUnderROC")

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Build the grid of different parameters
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.maxBins, [15, 20, 60]) \
    .build()

# Build out the cross validation
crossval = CrossValidator(estimator = dt,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluatorPR,
                          numFolds = 3)  

pipelineCV = Pipeline(stages=[indexer, va, crossval])

# Train the model using the pipeline, parameter grid, and preceding BinaryClassificationEvaluator
cvModel_u = pipelineCV.fit(train_b)

# COMMAND ----------

train_pred = cvModel_u.transform(train_b)
test_pred = cvModel_u.transform(test)
pr_train = evaluatorPR.evaluate(train_pred)
auc_train = evaluatorAUC.evaluate(train_pred)

# COMMAND ----------

# Create confusion matrix template
from pyspark.sql.functions import lit, expr, col, column

# Confusion matrix template
cmt = spark.createDataFrame([(1, 0), (0, 0), (1, 1), (0, 1)], ["Fraude", "prediction"])
cmt.createOrReplaceTempView("cmt")

# COMMAND ----------

# Source code for plotting confusion matrix is based on `plot_confusion_matrix` 
# via https://runawayhorse001.github.io/LearningApacheSpark/classification.html#decision-tree-classification
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, title):
  # Clear Plot
  plt.gcf().clear()

  # Configure figure
  fig = plt.figure(1)
  
  # Configure plot
  classes = ['Fraud', 'No Fraud']
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  # Normalize and establish threshold
  normalize=False
  fmt = 'd'
  thresh = cm.max() / 2.

  # Iterate through the confusion matrix cells
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  # Final plot configurations
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label') 
  
  # Display images
  image = fig
  
  # Show plot
  #fig = plt.show()
  
  # Save plot
  fig.savefig("confusion-matrix.png")

  # Display Plot
  display(image)
  
  # Close Plot
  plt.close(fig)

# COMMAND ----------

# Create temporary view for test predictions
test_pred.createOrReplaceTempView("test_pred")

Test_pred_cmdf = spark.sql("select a.Fraude, a.prediction, coalesce(b.count, 0) as count from cmt a left outer join (select Fraude, prediction, count(1) as count from test_pred group by Fraude, prediction) b on b.Fraude = a.Fraude and b.prediction = a.prediction order by a.Fraude desc, a.prediction desc")

# View confusion matrix
display(Test_pred_cmdf)

# COMMAND ----------

cm_pdf = Test_pred_cmdf.toPandas()

# Create 1d numpy array of confusion matrix values
cm_1d = cm_pdf.iloc[:, 2]

# Create 2d numpy array of confusion matrix values
cm = np.reshape(cm_1d, (-1, 2))

# Print out the 2d array
print(cm)

# COMMAND ----------

plot_confusion_matrix(cm, "Confusion Matrix ")

# COMMAND ----------

train_pred = Model_b.transform(train_b)
test_pred = Model_b.transform(test)
pr_train = evaluatorPR.evaluate(train_pred)
auc_train = evaluatorAUC.evaluate(train_pred)

# Evaluate the model on training datasets
pr_train = evaluatorPR.evaluate(train_pred)
auc_train = evaluatorAUC.evaluate(train_pred)

# Evaluate the model on test datasets
pr_test = evaluatorPR.evaluate(test_pred)
auc_test = evaluatorAUC.evaluate(test_pred)

# Print out the PR and AUC values
print("PR train:", pr_train)
print("AUC train:", auc_train)
print("PR test:", pr_test)
print("AUC test:", auc_test)

# COMMAND ----------

import mlflow
import mlflow.spark

import os
mlflow_experiment_id = 1030082808670079
with mlflow.start_run(experiment_id = mlflow_experiment_id) as run:
  # Log Parameters and metrics
  mlflow.log_param("balanced", "yes")
  mlflow.log_metric("PR train", pr_train)
  mlflow.log_metric("AUC train", auc_train)
  mlflow.log_metric("PR test", pr_test)
  mlflow.log_metric("AUC test", auc_test)
  
  # Log model
  mlflow.spark.log_model(Model_b, "model")
  
  # Log Confusion matrix
  mlflow.log_artifact("confusion-matrix.png")

# COMMAND ----------

