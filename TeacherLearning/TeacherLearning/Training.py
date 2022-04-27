import multiprocessing
import TrainingModel, Utils
import tensorflow as tf
import transformers
import numpy as np
import tqdm
import os
import json
import pandas as pd

############################################Added by Yasmine############################################

def prepare_embeddings(embeddings_path):
    '''
    embeddings_path(str) = "/mnt/localdata/karoui/datasets/nlvr2/blip_text_embeddings_en/"
    
    '''
    filenames = {'train':'features_train.csv','val':'features_dev.csv','test':'features_test.csv'}
    embeddings = []
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(embeddings_path,filenames[split]), header=None)
        df.drop(0, inplace=True, axis=1)
        embeddings.append(np.array(df))
    result_np = np.concatenate((embeddings[0],embeddings[1] , embeddings[2]), axis=0)
    return result_np.tolist()

def prepare_sentences(annotation_path, language):
    '''
    annotation_path(str) : "/mnt/localdata/karoui/datasets/nlvr2/annotations/"
    language(str) : in ['id', 'sw', 'ta', 'tr', 'zh']
    
    '''
    filenames = {'train':f'nlvr_train_{language}.json','val':f'nlvr_dev_{language}.json','test':f'nlvr_test_{language}.json'}
    sentences = []
    for split in ['train', 'val', 'test']:
        annotations = json.load(open(os.path.join(annotation_path,filenames[split]),'r'))
        sentences_split = [ann['sentence'] for ann in annotations]
        sentences.append(sentences_split)
    result_np = np.concatenate((sentences[0],sentences[1] , sentences[2]), axis=0) 
    return result_np.tolist()
    
########################################################################################################   
def prepareDataset(tokenizer, numValidationSamples, annotation_path, embeddings_path, language):
    # This part you need to prepare yourself!
    # What is needed here is a list of sentences in whatever language(s) you are interested in
    # and a matching set of Clip-Text encoder embeddings for the English counter part.

    # Pre-computed CLIP-Text encoder embeddings for 2 Million images, can be found here:
    # https://drive.google.com/drive/folders/1I9a7naSZubUATWzLFv61DQMWyFlF7wR5
    
    print('Prepare sentences')
    sentences = prepare_sentences(annotation_path, language)
    print('Prepare embeddings')
    emeddings = prepare_embeddings(embeddings_path)
    print("Number of total training samples:", len(sentences))


    
    inSents, embs = shuffleData(sentences, emeddings)  # Shuffle before selecting validation data
    numTrainSamples= len(inSents)-numValidationSamples
    #trainSents, trainEmbs = inSents[numValidationSamples:], embs[numValidationSamples:]
    #evalSents, evalEmbs = inSents[:numValidationSamples], embs[:numValidationSamples]
    evalSents, evalEmbs = inSents[numTrainSamples:], embs[numTrainSamples:]
    trainSents, trainEmbs = inSents[:numTrainSamples], embs[:numTrainSamples]
    evalIds, evalAtt = Utils.batchEncode(evalSents, tokenizer)
    evalInData, evalLabels = (evalIds, evalAtt), tf.convert_to_tensor(evalEmbs, tf.float32)
    print("Number of training samples:", len(trainSents))
    print("Number of validation samples:", len(evalSents))

    return trainSents, trainEmbs, evalInData, evalLabels


def shuffleData(sents, embs):
    '''
    shuffleOrder = np.random.choice(range(len(sents)), len(sents), replace=False)
    f = lambda x: [x[i] for i in shuffleOrder]
    return f(sents), f(embs)
    '''
    return sents, embs


def createModel(modelBase, clipEmbeddingSize):
    model = TrainingModel.SentenceModelWithLinearTransformation(modelBase, clipEmbeddingSize)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelBase)
    return model, tokenizer


def trainStudentTextEncoder():
    #modelBase = 'distilbert-base-multilingual-cased'
    modelBase = "dbmdz/bert-base-turkish-cased"
    language = 'tr' #turkish
    annotation_path = "/mnt/localdata/karoui/datasets/nlvr2/annotations/"
    embeddings_path = "/mnt/localdata/karoui/datasets/nlvr2/blip_text_embeddings_en/"
    
    numValidationSamples = 2000
    clipEmbeddingSize = 768
    learningRate = 5e-5
    batchSize = 1
    epochs = 100
    fetchSize = 1 #500 * batchSize

    model, tokenizer = createModel(modelBase, clipEmbeddingSize)
    trainSents, trainEmbs, evalIn, evalLabels = prepareDataset(tokenizer, numValidationSamples, annotation_path, embeddings_path, language)

    optim = tf.optimizers.Adam(learningRate)
    model.compile(optim, loss='mse', metrics=['mae'])
    saveName = f"CLIP-Text-Encoder-{language}"

    fetchCounter = 0
    for e in range(epochs):
        shuffleData(trainSents, trainEmbs)
        for i in tqdm.tqdm(range(0, len(trainSents), fetchSize), desc="Fetches"):
            batchEmbs = tf.convert_to_tensor(trainEmbs[i:i + fetchSize], tf.float32)
            batchSents = trainSents[i:i + fetchSize]

            inData = Utils.batchEncode(batchSents, tokenizer)

            model.fit(inData, batchEmbs, batch_size=batchSize, verbose=1,
                      validation_data=(evalIn, evalLabels), shuffle=True)

            fetchCounter += 1
            model.save_pretrained("{}-{}-Weights".format(saveName, fetchCounter))
            #if (fetchCounter % 10 == 0):
                #model.save_weights("{}-{}-Weights".format(saveName, fetchCounter))
                #model.save_pretrained("{}-{}-Weights".format(saveName, fetchCounter))
                


if __name__ == '__main__':
    
    trainStudentTextEncoder()
