# Named Entity Recognition — Simple Transformers —Flask REST API

Medium article [here](https://medium.com/swlh/named-entity-recognition-simple-transformers-flask-rest-api-ec14a7a444cb)!

Named-entity recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes.

I wanted to start with NER, I looked at SpaCy and nltk. SpaCy already has a pre trained NER model, which can be custom trained. Started with that approach. Then I decided to explore transformers in NLP.

I looked for something which can be implemented fast and found out the amazing Simple Transformers library created by Thilina Rajapakse.

Simple Transformers lets you quickly train and evaluate Transformer models.The Simple Transformers library is built on top of the Transformers library by Hugging Face.

### The process of Named Entity Recognition has four steps:

  - Obtaining training and testing data
  - Preprocessing the data into dataframes required for the model inputs
  - Training the model
  - Testing the model
  - Tuning the model further for better accuracy
  - Creating an interface for the model

Obtaining training and testing data

Regarding obtaining the data, the coNLL data set can be used to start with. I was given the data from a company for a project, so the data isn’t publicly available.

So here is the format we need the data in: A list of lists with three columns, sentence number, word and the tag. Same for the test data as well.

![image](https://user-images.githubusercontent.com/51290447/112844131-01e7fa00-90c1-11eb-909c-81d08b8de43f.png)


I tried it in google colab, and connected drive for getting the data. Here is the format my dataset was in, which if a real life dataset is constructed manually will look like.

My raw data was in this format :

```
Text : sentence
Rest unnamed columns :tags corresponding to each tag in the sentence
```
![image](https://user-images.githubusercontent.com/51290447/112844167-0f9d7f80-90c1-11eb-937c-f9949bf52349.png)


## Preprocessing the data into dataframes required for the model inputs :

To parse the data in the format required by simple transformers, I divided the data in 85:15 ratio and wrote a parser.

```
import pandas as pd
df=pd.read_csv("drive/My Drive/Data.csv")
Sentences=df['Text']
df['Labels'] = df['Unnamed: 1'].astype(str) +" "+ df['Unnamed: 2'].astype(str)+" "+ df['Unnamed: 3'].astype(str)+" "+ df['Unnamed: 4'].astype(str)+" "+ df['Unnamed: 5'].astype(str)+" "+ df['Unnamed: 6'].astype(str)+" "+ df['Unnamed: 7'].astype(str)+" "+ df['Unnamed: 8'].astype(str)+" "+ df['Unnamed: 9'].astype(str)+" "+ df['Unnamed: 10'].astype(str)+" "+ df['Unnamed: 11'].astype(str)
df=df.drop(columns=['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4',
       'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
       'Unnamed: 10', 'Unnamed: 11'],axis=1)
Labels=df['Labels']
traindata=[]
for i in range(0,len(Sentences[:95])):
  wordslist=Sentences[i].split(" ")
  labelslist=Labels[i].split(" ")
  

  for j in range(0,len(wordslist[:95])):
    traindata.append([i,wordslist[j],labelslist[j]])
testdata=[]
for i in range(95,105):
  wordslist=Sentences[i].split(" ")
  labelslist=Labels[i].split(" ")
  

  for j in range(0,len(wordslist)):
    testdata.append([i-95,wordslist[j],labelslist[j]])
```

df: dataframe loads the data CSV file into a pandas dataframe.
Sentences : List to store the Text column of the dataframe
df['Labels'] : New column in dataframe to add the strings of all tags in the unnamed columns of the dataframe
Labels : List to store the Labels column of the dataframe
traindata : List of lists with each list: [sentence number,word,tag] from 0th sentence to 94th sentence.
testdata : List of lists with each list: [sentence number,word,tag] from 95th sentence to 106th sentence.

## Training the model

Training the model is real easy, though it may take some time.

```
from simpletransformers.ner import NERModel
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
train_df = pd.DataFrame(traindata, columns=['sentence_id', 'words', 'labels'])
test_df = pd.DataFrame(testdata, columns=['sentence_id', 'words', 'labels'])
model = NERModel('bert', 'bert-base-cased', 
                 labels=[ENTER YOUR LIST OF UNIQUE LABELS IN YOUR DATA],
args={"save_eval_checkpoints": False,
      "save_steps": -1,
      "output_dir": "drive/My Drive/MODEL",
      'overwrite_output_dir': True,
      "save_model_every_epoch": False,
      'reprocess_input_data': True, 
      "train_batch_size": 10,'num_train_epochs': 5,"max_seq_length": 256, "gradient_accumulation_steps": 8}, use_cuda=False)
model.train_model(train_df)
```

The first trial gave very low f1 score, that was because by default the number of epochs was set to one.

Here I used the bert-base-cased model to train on as my data was cased. Model is saved to MODEL folder so that we do not need to train the model everytime to test it. We can load the trained model directly from the folder.

```
train_batch_size : Batch size, preferably a power of 2 gives good results
```

## Testing the model

Now check the outputs by passing any arbitary sentence :

```
result, model_outputs, predictions = model.eval_model(test_df)
predictions, raw_outputs = model1.predict(["Tax Rate"])
print(predictions)
```

In place of ‘tax rate’, any arbitrary sentence can be entered to check output tags.
Tuning the model further for better accuracy

Now lets try tuning the Hyper Parameters (args). These are the args that are available :

![image](https://user-images.githubusercontent.com/51290447/112844513-715de980-90c1-11eb-8f9b-8290029250fc.png)


https://pypi.org/project/simpletransformers/

Here is the link to all the hyperparameters.

Increasing the number of epochs and max_seq_length helped in getting the accuracy to 86%. By using a large model (roBERTa large), I was able to achieve 92.6% accuracy on the test data.

Outputs:

![image](https://user-images.githubusercontent.com/51290447/112844563-7fac0580-90c1-11eb-849f-f7e2866a6213.png)

```
INFO:simpletransformers.ner.ner_model:{'eval_loss': 0.6981344372034073, 'precision': 0.9069767441860465, 'recall': 0.9512195121951219, 'f1_score': 0.9285714285714286}
```

There are several pretrained models that can be used. You can choose one depending on whether your data is in a specific language, or if your data is cased. My training data was financial and in English. So BERT and roBERTa were my best options. I did try distilBERT as well.

Supported model types for Named Entity Recognition in Simple Transformers:

   - BERT
   - CamemBERT
   - DistilBERT
   - ELECTRA
   - RoBERTa
   - XLM-RoBERTa

## Creating an interface for the model

I worked on creating an interface for the model. Using Flask and Rest API, I created a web page with two buttons, one to display the results of the sentence entered (GET NER TAGS).

Here is the Flask app.py code :

```
import numpy as np
from flask import Flask, request, jsonify, render_template
import simpletransformers
import pandas as pd
import requests
from simpletransformers.ner import NERModel
import json 
import io
from flask_csv import send_csv
from flask import make_response
from flask import session, redirect

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST',"GET"])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    sentence = int_features[0]
    if int_features[1] == 'display':
        model1 = NERModel('bert', 'MODEL',
                  labels=[LIST OF UNIQUE LABELS],
                  args={"save_eval_checkpoints": False,
        "save_steps": -1,
        "output_dir": "MODEL",
        'overwrite_output_dir': True,
        "save_model_every_epoch": False,
        'reprocess_input_data': True, 
        "train_batch_size": 10,'num_train_epochs': 5,"max_seq_length": 256}, use_cuda=False)
        predictions, raw_outputs = model1.predict([sentence])     
        result = json.dumps(predictions[0])        
        return render_template('index.html', prediction_text=result)

    elif int_features[1] == 'getcsv':
        model1 = NERModel('bert', 'MODEL',
                  labels=[LIST OF LABELS],
                  args={"save_eval_checkpoints": False,
        "save_steps": -1,
        "output_dir": "MODEL",
        'overwrite_output_dir': True,
        "save_model_every_epoch": False,
        'reprocess_input_data': True, 
        "train_batch_size": 10,'num_train_epochs': 5,"max_seq_length": 256}, use_cuda=False)
        predictions, raw_outputs = model1.predict([sentence])
        l=[]
        for i in predictions[0]:
            dic={}
            for j in i.keys():
                dic['word']=j
                dic['tag']=i[j]
            l.append(dic)
        return send_csv(l,"tags.csv",["word","tag"])
      
if __name__ == "__main__":
    app.run(debug=True)
```

Make sure you use the same args to load the model as used to train the model.

When you click on the (GET CSV) button, a csv of the words in the sentence and the respective tags gets downloaded automatically.

![image](https://user-images.githubusercontent.com/51290447/112844697-a4a07880-90c1-11eb-8d56-f14248e95ad4.png)


Structure of tags.csv:

![image](https://user-images.githubusercontent.com/51290447/112844710-ab2ef000-90c1-11eb-9514-168672446180.png)


Please leave a star if you found this useful. Thanks!
