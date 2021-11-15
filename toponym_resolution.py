#pip install geopandas

#pip install rasterio

#pip install healpy

#pip install pyproj

#pip install transformers

import os
import sys
import math
import time
import pyproj
import datetime
import argparse
import healpy
import rasterio
import random
import numpy as np
import gc
import tensorflow as tf
import torch
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, DebertaTokenizer, ConvBertTokenizer, MPNetTokenizer
from itertools import chain
from itertools import repeat
from itertools import combinations
from geopy import distance
from math import radians, cos, sin, asin, sqrt
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from shapely.geometry import box
from fiona.crs import from_epsg
from rasterio.mask import mask

geoproperties = 1
if geoproperties == 1:
    import warnings
    warnings.filterwarnings("ignore")
ce_weight = 1
gcd_weight = 0.0005
cl_weight = 0.0001
lc_weight = 0.01
el_weight = 0.1
ve_weight = 0.1
c_loss = 0 #0 - contrastive loss is not considered | 1 - contrastive loss is considered
input_data = 2 #0-WOTR | 1-LGL | 2-SpatialML | 3-SemEval 
LINES=8500    #number of lines I am considering in a document | If the all document is to be considered then LINES = 8500
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)

if input_data == 0:
    input_trainfile_path = "/content/drive/MyDrive/TR/corpora/WOTR/wotr-train.txt"
    input_testfile_path = "/content/drive/MyDrive/TR/corpora/WOTR/wotr-test.txt"
elif input_data == 1:
    input_trainfile_path = "/content/drive/MyDrive/TR/corpora/LGL/lgl-train.txt"
    input_testfile_path = "/content/drive/MyDrive/TR/corpora/LGL/lgl-test.txt"
elif input_data == 2:
    input_trainfile_path = "/content/drive/MyDrive/TR/corpora/SpatialML/spatialML-train.txt"
    input_testfile_path = "/content/drive/MyDrive/TR/corpora/SpatialML/spatialML-test.txt"
elif input_data == 3:
    input_trainfile_path = "/content/drive/MyDrive/TR/corpora/SemEval19-Task12-Challenge/semeval-train-90.txt"
    input_testfile_path = "/content/drive/MyDrive/TR/corpora/SemEval19-Task12-Challenge/semeval-test-10.txt"
else:
    input_trainfile_path = "/content/drive/MyDrive/TR/corpora/GeoVirus/geovirus-train-1.txt"
    input_testfile_path = "/content/drive/MyDrive/TR/corpora/GeoVirus/geovirus-test-1.txt"

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    print("GPU device not found")
    #raise SystemError('GPU device not found')

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#Utils file

def newsplit(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])

def latlon2healpix( lat , lon , res ):
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0
    xs = ( math.cos(lat) * math.cos(lon) )
    ys = ( math.cos(lat) * math.sin(lon) )
    zs = ( math.sin(lat) )
    return healpy.vec2pix( int(res) , xs , ys , zs )

def healpix2latlon( code , res ):
    [xs, ys, zs] = healpy.pix2vec( int(res) , code )
    lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
    lon = float( math.atan2(ys, xs) * 180.0 / math.pi )
    return [ lat , lon ]

def getCoordinatesFromTrainAndTest (trainfilename, testfilename):
    trainCoords = []
    testCoords = []
    with open(trainfilename) as f:
        lines = f.readlines()
        i=0
        for line in lines:
            i+=1
            x = line.split("[",1)[1]
            y = x.split("]",1)[0]
            if y.find(",") != -1:
                lat = y.split(",")[0]
                long = y.split(",")[1]
            else:
                newx = newsplit(line,"[",2)[1]
                newy = newx.split("]",1)[0]
                lat = newy.split(",")[0]
                long = newy.split(",")[1]
            trainCoords.append([float(lat),float(long)])
    with open(testfilename) as f:
        lines = f.readlines()
        i=0
        for line in lines:
            i+=1
            x = line.split("[",1)[1]
            y = x.split("]",1)[0]
            if y.find(",") != -1:
                lat = y.split(",")[0]
                long = y.split(",")[1]
            else:
                newx = newsplit(line,"[",2)[1]
                newy = newx.split("]",1)[0]
                lat = newy.split(",")[0]
                long = newy.split(",")[1]
            testCoords.append([float(lat),float(long)])

    return trainCoords, testCoords

def getCoordinatesFromTrain(trainfilename):
    trainCoords = []
    with open(trainfilename) as f:
        lines = f.readlines()
        i=0
        for line in lines:
            i+=1
            x = line.split("[",1)[1]
            y = x.split("]",1)[0]
            if y.find(",") != -1:
                lat = y.split(",")[0]
                long = y.split(",")[1]
            else:
                newx = newsplit(line,"[",2)[1]
                newy = newx.split("]",1)[0]
                lat = newy.split(",")[0]
                long = newy.split(",")[1]
            if float(lat) > 90 or float(lat) < -90:
                print(i)
            trainCoords.append([float(lat),float(long)])

    return trainCoords

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
def trainingValidationSplit (train, size):
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train[0], train[2], random_state=seed_val, test_size=size)

    train_masks, validation_masks, _, _ = train_test_split(train[1], train[2], random_state=seed_val, test_size=size)

    train_land_coverage, validation_land_coverage, _ , _ = train_test_split(train[3], train[2], random_state= seed_val, test_size=size)
    train_elevation, validation_elevation, _, _ = train_test_split(train[4], train[2], random_state= seed_val, test_size=size)
    train_vegetation, validation_vegetation, _, _ = train_test_split(train[5], train[2], random_state=seed_val, test_size=size)

    train = [train_inputs, train_masks, train_labels, train_land_coverage, train_elevation, train_vegetation]
    validation = [validation_inputs, validation_masks, validation_labels, validation_land_coverage, validation_elevation, validation_vegetation]
    return train, validation

def getLatLong(codes, res):
    coords = []
    for code in codes:
        aux = healpy.vec2pix(res, code[0], code[1], code[2])
        coords.append(healpix2latlon(aux, res))
    return coords


    
def great_circle_distance( p1 , p2 , transf=None):
    if transf == "both":
        aa0 = torch.atan2(p1[:,2], torch.sqrt(p1[:,0] ** 2 + p1[:,1] ** 2)) * 180.0 / 3.141592653589793238462643383279502884197169399375105820974944592307816406286
        aa1 = torch.atan2(p1[:,1], p1[:,0]) * 180.0 / 3.141592653589793238462643383279502884197169399375105820974944592307816406286
        aa0 = aa0 * 0.01745329251994329576924
        aa1 = aa1 * 0.01745329251994329576924  
    else:
        aa0 = p1[:,0] * 0.01745329251994329576924
        aa1 = p1[:,1] * 0.01745329251994329576924
    if transf == "both" or transf == "onlypredicted":
        bb0 = torch.atan2(p2[:,2], torch.sqrt(p2[:,0] ** 2 + p2[:,1] ** 2)) * 180.0 / 3.141592653589793238462643383279502884197169399375105820974944592307816406286
        bb1 = torch.atan2(p2[:,1], p2[:,0]) * 180.0 / 3.141592653589793238462643383279502884197169399375105820974944592307816406286
        bb0 = bb0 * 0.01745329251994329576924
        bb1 = bb1 * 0.01745329251994329576924
    else:
        bb0 = p2[:,0] * 0.01745329251994329576924
        bb1 = p2[:,1] * 0.01745329251994329576924
    sin_lat1 = torch.sin( aa0 )
    cos_lat1 = torch.cos( aa0 )
    sin_lat2 = torch.sin( bb0 )
    cos_lat2 = torch.cos( bb0 )
    delta_lng = bb1 - aa1
    cos_delta_lng = torch.cos(delta_lng)
    sin_delta_lng = torch.sin(delta_lng)
    d = torch.atan2(torch.sqrt((cos_lat2 * sin_delta_lng) ** 2 + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2), sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng )
    return torch.mean( 6371.0087714 * d , axis = -1 )

def cube(vector):
    new_vector = (vector**3.0)/(torch.sum(vector**3.0))
    return new_vector

def geodistance_vincenty( coords1 , coords2 ):
    lat1 , lon1 = coords1[ : 2]
    lat2 , lon2 = coords2[ : 2]

    try: return distance.vincenty( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0
    except: return distance.great_circle( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0
  

def calculateDistances(truth_coordinates, predicted_coordinates):
    distances = []
    accAt161 = 0
    for i in range(0, len(truth_coordinates)):
        distance = geodistance_vincenty(truth_coordinates[i], predicted_coordinates[i])
        distances.append(distance)
        if distance < 161:
            accAt161 += 1
    return distances , accAt161


def calculateContrastiveLoss(truth_coordinates, predicted_coordinates):
    combs = list(combinations(range(len(truth_coordinates)), 2))
    aux = 0
    for i in range(0, len(combs)):
        comb_left = combs[i][0]
        comb_right = combs[i][1]
        ground_truth_left = torch.reshape(truth_coordinates[comb_left], (1,2))
        ground_truth_right = torch.reshape(truth_coordinates[comb_right], (1,2))
        ground_truth_distance = great_circle_distance(ground_truth_left, ground_truth_right)

        predicted_left = torch.reshape(predicted_coordinates[comb_left], (1,3))
        predicted_right = torch.reshape(predicted_coordinates[comb_right], (1,3))
        predicted_distance = great_circle_distance(predicted_left, predicted_right, "both")
        if ground_truth_distance < 161:
            if aux == 0:
                ground_truth = torch.reshape(ground_truth_distance, (1,1))
                predicted = torch.reshape(predicted_distance, (1,1))
                aux += 1
            else:
                ground_truth_distance_reshaped = torch.reshape(ground_truth_distance, (1,1))
                predicted_distance_reshaped = torch.reshape(predicted_distance, (1,1))

                ground_truth = torch.cat((ground_truth, ground_truth_distance_reshaped), 0)
                predicted = torch.cat((predicted, predicted_distance_reshaped), 0)
        else:
            if i == len(combs)-1 and aux==0:
                dummy_ground_truth = torch.tensor(0, dtype=float)
                dummy_predicted = torch.tensor(0, dtype=float)
                ground_truth = torch.reshape(dummy_ground_truth, (1,1))
                predicted = torch.reshape(dummy_predicted, (1,1))
    return ground_truth, predicted
        
def flatten(aux):
    flat_list = []
    for sublist in aux:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def calculateLoss(predicted_codes, ground_truth_coordinates, model_outputs, labels, land_coverage, land_coverage_codes, elevation, elevation_codes, vegetation, vegetation_codes):
    great_circle = great_circle_distance(ground_truth_coordinates, predicted_codes, "onlypredicted")
    crossEntropy_loss = nn.CrossEntropyLoss()(model_outputs, torch.max(labels,1)[1])
    #criterion = ASLSingleLabel()
    #crossEntropy_loss = criterion(model_outputs, torch.max(labels,1)[1])
    land_coverage_loss = nn.CrossEntropyLoss()(land_coverage_codes, torch.max(land_coverage,1)[1])
    elevation_loss = nn.L1Loss()(elevation_codes, elevation)
    vegetation_loss = nn.L1Loss()(vegetation_codes, vegetation)
    if c_loss == 1:
        ground_truth_distances, predicted_distances = calculateContrastiveLoss(ground_truth_coordinates, predicted_codes)
        contrastive_loss = nn.MSELoss()(predicted_distances, ground_truth_distances)
        contrastive_loss_sqrt = torch.sqrt(contrastive_loss)
        loss = ce_weight*crossEntropy_loss + gcd_weight*great_circle + cl_weight*contrastive_loss_sqrt
    else:
        loss = ce_weight*crossEntropy_loss + gcd_weight*great_circle + lc_weight*land_coverage_loss + el_weight*elevation_loss + ve_weight*vegetation_loss
    return loss


def calculateBatchMetrics(predicted_codes, ground_truth_coordinates, model_outputs, labels):

    b = np.zeros_like(model_outputs)
    b[np.arange(len(model_outputs)), model_outputs.argmax(1)] = 1
    res = (labels[:, None] == b).all(-1).any(-1)
    coords = getLatLong(predicted_codes, 256)
    vincenty, accAt161 = calculateDistances(ground_truth_coordinates, coords)
    return sum(res), vincenty, accAt161



def get_values(lc, el, ve, bbox):
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
    
    geo_lc = geo.to_crs(crs=lc.crs.data)
    geo_el = geo.to_crs(crs=el.crs.data)
    geo_ve = geo.to_crs(crs=ve.crs.data)

    coords_lc = getFeatures(geo_lc)
    coords_el = getFeatures(geo_el)
    coords_ve = getFeatures(geo_ve)

    out_img_lc, _ = mask(dataset=lc, shapes=coords_lc, crop=True)
    out_img_el, _ = mask(dataset=el, shapes=coords_el, crop=True)
    out_img_ve, _ = mask(dataset=ve, shapes=coords_ve, crop=True)

    out_img_el[0][out_img_el[0] < -100] = 0.0
    out_img_ve[0][out_img_ve[0] > 100] = 0.0

    el_v = out_img_el[0].mean()
    ve_v = out_img_ve[0].mean()

    matrix = []
    for i in range(0, len(out_img_lc[0])):
        a = to_categorical(out_img_lc[0][i], num_classes=21)
        for j in a:
            matrix.append(j)
    new_matrix = np.array(matrix)
    column_sums = new_matrix.sum(axis=0)
    lc_v = [x / len(matrix) for x in column_sums]

    ve_v = ve_v / 100.0
    el_v = el_v / 8850.0

    if el_v < 0: el_v = 0.0
    if ve_v < 0: ve_v = 0.0

    return [lc_v, np.array([ el_v, ve_v])]

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

import torch.nn as nn
import transformers
class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

class LanguageModel (nn.Module):
    def __init__ (self, num_classes):
        super(LanguageModel, self).__init__()
        #self.bert = transformers.BertModel.from_pretrained("bert-large-uncased", return_dict=False)
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        #self.bert = transformers.ConvBertModel.from_pretrained('YituTech/conv-bert-base', return_dict=False)
        #self.bert = transformers.MPNetModel.from_pretrained('microsoft/mpnet-base', return_dict=False)
        #self.bert = transformers.RobertaModel.from_pretrained('roberta-large', return_dict=False)
        self.drop = nn.Dropout(0.3)
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Linear(2304, self.num_classes)
        #self.out = nn.Linear(3072, self.num_classes)
    
    def forward(self, top_input_ids, top_attention_masks, close_input_ids, close_attention_masks, wide_input_ids, wide_attention_masks):
        t_outputs = self.bert(top_input_ids, attention_mask=top_attention_masks, token_type_ids=None)
        t_cls = t_outputs[0][:,0:1,:]
        t_reshaped = t_cls.reshape(len(t_outputs[0]), 768)
        #t_reshaped = t_cls.reshape(len(t_outputs[0]), 1024)
        c_outputs = self.bert(close_input_ids, attention_mask=close_attention_masks, token_type_ids=None)
        c_cls = c_outputs[0][:,0:1,:]
        c_reshaped = c_cls.reshape(len(c_outputs[0]), 768)
        #c_reshaped = c_cls.reshape(len(c_outputs[0]), 1024)
        w_outputs = self.bert(wide_input_ids, attention_mask=wide_attention_masks, token_type_ids=None)
        w_cls = w_outputs[0][:,0:1,:]
        w_reshaped = w_cls.reshape(len(w_outputs[0]), 768)
        #w_reshaped = w_cls.reshape(len(w_outputs[0]), 1024)
        top_outputs = self.drop(t_reshaped)
        close_outputs = self.drop(c_reshaped)
        wide_outputs = self.drop(w_reshaped)
        res = torch.cat([top_outputs, close_outputs, wide_outputs],1)
        outputs = self.out(res)
        return outputs

    def normalize(self, outputs):
        return self.softmax(outputs)



def trainLoopFunction (top_data_loader, close_data_loader, wide_data_loader, coords_data_loader, model, optimizer, device, codes_matrix, land_coverage_matrix, elevation_matrix, vegetation_matrix, scheduler=None):
    model = model.to(device)
    model.train()
    total_loss = 0
    total_class_accuracy = 0
    totalAt161 = 0
    totalDistances = []
    #optimizer.zero_grad()
    for i, (d, e, f, g) in enumerate(zip(top_data_loader, close_data_loader, wide_data_loader, coords_data_loader)):

        top_ids = d[0]
        top_masks = d[1]
        labels = d[2]
        land_coverage = d[3]
        elevation = d[4]
        vegetation = d[5]
        bsize = len(top_ids)
        close_ids = e[0]
        close_masks = e[1]
        wide_ids = f[0]
        wide_masks = f[1]
        coordinates = g[0]
        top_ids = top_ids.to(device, dtype=torch.long)
        top_masks = top_masks.to(device, dtype=torch.bool)
        labels = labels.to(device, dtype=torch.float)
        land_coverage = land_coverage.to(device, dtype=torch.float)
        elevation = elevation.to(device, dtype=torch.float)
        vegetation = vegetation.to(device, dtype=torch.float)
        close_ids = close_ids.to(device, dtype=torch.long)
        close_masks = close_masks.to(device, dtype=torch.bool)
        wide_ids = wide_ids.to(device, dtype=torch.long)
        wide_masks = wide_masks.to(device, dtype=torch.bool)      
        coordinates = coordinates.to(device, dtype=torch.float)
        codes = codes_matrix.to(device, dtype=torch.float)
        land_coverage_mat = land_coverage_matrix.to(device, dtype=torch.float)
        elevation_mat = elevation_matrix.to(device, dtype=torch.float)
        vegetation_mat = vegetation_matrix.to(device, dtype=torch.float)
        #torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        outputs = model(top_ids, top_masks, close_ids, close_masks, wide_ids, wide_masks)
        normalized_outputs = model.normalize(outputs)
        cube_outputs = cube(normalized_outputs)
        region_codes = torch.matmul(cube_outputs, codes)
        land_coverage_codes = torch.matmul(cube_outputs, land_coverage_mat)
        elevation_codes = torch.matmul(cube_outputs, elevation_mat)
        vegetation_codes = torch.matmul(cube_outputs, vegetation_mat)
        loss = calculateLoss(region_codes, coordinates, outputs, labels, land_coverage, land_coverage_codes, elevation, elevation_codes, vegetation, vegetation_codes )
        total_loss += loss
        #loss = loss / 4
        loss.backward()
        #Gradient Acccumulation
        """
        if (i+1)%4 == 0:
            optimizer.step()       
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        """

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        #Calculate class accuracy 
        logits = outputs.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        region_codes = region_codes.detach().cpu().numpy()
        coordinates = coordinates.detach().cpu().numpy()

        
        res, vincenty, accAt161 = calculateBatchMetrics(region_codes, coordinates, logits, label_ids)
        total_class_accuracy += res
        totalAt161 += accAt161
        totalDistances.append(vincenty)

        del outputs
        gc.collect()
        del labels
        gc.collect()
        del land_coverage
        gc.collect()
        del vegetation
        gc.collect()
        del elevation
        gc.collect()
        del region_codes
        gc.collect()
        del land_coverage_mat
        gc.collect()
        del land_coverage_codes
        gc.collect()
        del elevation_codes
        gc.collect()
        del elevation_mat
        gc.collect()
        del vegetation_codes
        gc.collect()
        del vegetation_mat
        gc.collect()
        del vincenty
        gc.collect()
        del accAt161
        gc.collect()
        del top_ids
        gc.collect()
        del top_masks
        gc.collect()
        del close_ids
        gc.collect()
        del close_masks
        gc.collect()
        del wide_ids
        gc.collect()
        del wide_masks
        gc.collect()
        del normalized_outputs
        gc.collect()
        del cube_outputs
        gc.collect()
        del coordinates
        gc.collect()
        del codes 
        gc.collect()
        del label_ids
        gc.collect()
        del logits
        gc.collect()
        torch.cuda.empty_cache()

    aux_total = flatten(totalDistances)
    median = np.median(aux_total)
    mean = np.mean(aux_total)
    
    del totalDistances
    gc.collect()
    del aux_total
    gc.collect()
    torch.cuda.empty_cache()

    return total_loss / len(top_data_loader), total_class_accuracy, mean, totalAt161, median


def evalLoopFunction (top_data_loader, close_data_loader, wide_data_loader, coords_data_loader, model, device, codes_matrix):
    model = model.to(device)
    model.eval()
    total_class_accuracy = 0
    totalAt161 = 0
    totalDistances = []

    for i, (d, e, f, g) in enumerate(zip(top_data_loader, close_data_loader, wide_data_loader, coords_data_loader)):

        top_ids = d[0]
        top_masks = d[1]
        labels = d[2]   
        close_ids = e[0]
        close_masks = e[1]
        wide_ids = f[0]
        wide_masks = f[1]
        coordinates = g[0]

        top_ids = top_ids.to(device, dtype=torch.long)
        top_masks = top_masks.to(device, dtype=torch.bool)
        labels = labels.to(device, dtype=torch.float)
        close_ids = close_ids.to(device, dtype=torch.long)
        close_masks = close_masks.to(device, dtype=torch.bool)
        wide_ids = wide_ids.to(device, dtype=torch.long)
        wide_masks = wide_masks.to(device, dtype=torch.bool)
        coordinates = coordinates.to(device, dtype=torch.float)
        codes = codes_matrix.to(device, dtype=torch.float)
        outputs = model(top_ids, top_masks, close_ids, close_masks, wide_ids, wide_masks)

        normalized_outputs = model.normalize(outputs)
        cube_outputs = cube(normalized_outputs)
        #interpolation
        region_codes = torch.matmul(cube_outputs, codes)
        region_codes = region_codes.detach().cpu().numpy()
        #get the coordinates from the region predicted
        coords = getLatLong(region_codes, 256)

        coordinates = coordinates.detach().cpu().numpy()
        
        #calculate distance accuracy
        vincenty, accAt161 = calculateDistances(coordinates, coords)
        
        totalAt161 += accAt161
        totalDistances.append(vincenty)

        #calculate class accuracy
        final_labels = labels.to('cpu').numpy()
        final_outputs=outputs.cpu().detach().numpy()
        b = np.zeros_like(final_outputs)
        b[np.arange(len(final_outputs)), final_outputs.argmax(1)] = 1
        res = (final_labels[:, None] == b).all(-1).any(-1)
        total_class_accuracy += sum(res)

        del outputs
        gc.collect()
        del labels
        gc.collect()
        del vincenty
        gc.collect()
        del accAt161
        gc.collect()
        del top_ids
        gc.collect()
        del top_masks
        gc.collect()
        del close_ids
        gc.collect()
        del close_masks
        gc.collect()
        del wide_ids
        gc.collect()
        del wide_masks
        gc.collect()
        del coordinates
        gc.collect()
        del codes 
        gc.collect()
        del b
        gc.collect()
        del final_labels
        gc.collect()
        del final_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
    aux_total = flatten(totalDistances)
    median = np.median(aux_total)
    mean = np.mean(aux_total)
    torch.cuda.empty_cache()

    del totalDistances
    gc.collect()
    del aux_total
    gc.collect()

    torch.cuda.empty_cache()
    return total_class_accuracy, mean, totalAt161, median

MAX_LEN = 512

def getBERTInputs(filename, labels, land_coverage, elevation, vegetation):
    #in this approach there will be 3 different inputs to the model:
    # - the input_ids and att_masks associated with the target toponym
    # - the input_ids and att_masks associated with the close context
    # - the input_ids and att_masks associated with the wide context
    input_ids_toponyms = []
    attention_toponyms = []

    input_ids_close = []
    attention_close = []
    labels_close = []
    
    elevation_close = []
    vegetation_close = []
    land_coverage_close = []

    input_ids_wider = []
    attention_wider = []
    #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    #tokenizer = ConvBertTokenizer.from_pretrained('YituTech/conv-bert-base', do_lower_case=True)
    #tokenizer = MPNetTokenizer.from_pretrained('microsoft/mpnet-base', do_lower_case=True)
    
    with open(filename) as f:
        aux=0
        lines = f.readlines()
        for line in lines[:LINES]:
            #auxiliary lists
            local_close_context = []
            local_wider_context = []
            local_toponym_mention = []
            local_input_ids_toponyms = []
            local_input_ids_close = []
            local_input_ids_wider = []

            #split each line of the document accordingly
            x = line.split("[",1)
            y = x[1].split("]",1)
            #if I find a comma aka coordinates
            if y[0].find(",") != -1:
                toponym = x[0]
                rest = x[1]
                text = rest.split("]", 1)[1]
            #if the toponym has [ or ]
            else:
                toponym = newsplit(line,"[",2)[0]
                rest = newsplit(line,"[",2)[1]
                text = rest.split("]",1)[1]

            #target toponym tokenized
            toponym_tokenization = tokenizer.tokenize(toponym)
            text_tokenization = tokenizer.tokenize(text)

            #print(aux)
            len_text = len(text_tokenization) 
            len_toponym = len(toponym_tokenization) 
            half_len_text = len_text // 2

            #where_toponym is the positions associated with the ocurrence of the target toponym in the text
            where_toponym = []
            #print(aux)
            for i in range(0,len(text_tokenization)):
                aux_j = i
                for j in range(0, len(toponym_tokenization)):
                    if toponym_tokenization[j] == text_tokenization[aux_j]:
                        if j+1 == len(toponym_tokenization):
                            where_toponym.append(i)
                        else:
                            aux_j+=1
                            pass
                    else:
                        break
            #I want the ocurrence closer to the middle of the text
            where_toponym = [min(where_toponym, key=lambda x:abs(x-half_len_text))]
            #Since the BERT model is limited to 512 tokens, and 2 of them are the CLS and SEP tokens, i can only consider 510 tokens at max
            #If the toponym has len=1, i want 255 tokens to the left of the toponym and 254 to the right, making it 510 total   
            big_left = 255
            big_right = 255
            small_left = 25
            small_right = 25
            #value updates
            for i in range(0, len_toponym):
                if (i+1) % 2 == 1:
                    small_right -= 1
                    big_right -= 1
                else:
                    small_left -= 1
                    big_left -= 1

            #for the close context   
            for i in where_toponym:
                res=[]
                #if the ocurrence is very close to the beginning of the text
                if i-small_left < 0:
                    res.append(text_tokenization[:i])
                    tokens_left = small_left-len(list(chain.from_iterable(res)))
                    if i+tokens_left+small_right > len_text:
                        res.append(text_tokenization[i:])
                    else:
                        res.append(text_tokenization[i:i+tokens_left+len_toponym+small_right])                  
                #if the ocurrence is very close to the end of the text
                elif i+len_toponym+small_right > len_text:
                    tokens_left = small_right-len(text_tokenization[i:])+len_toponym
                    if i-tokens_left-small_left < 0:
                        res.append(text_tokenization[:i])
                    else:
                        res.append(text_tokenization[i-small_left-tokens_left:i])
                    res.append(text_tokenization[i:])
                #in the perfect case
                else:
                    res.append(text_tokenization[i-small_left:i])
                    res.append(text_tokenization[i:i+len_toponym+small_right])

                local_close_context.append(list(chain.from_iterable(res)))
                land_coverage_close.append(land_coverage[aux])
                vegetation_close.append(vegetation[aux])
                elevation_close.append(elevation[aux])
                labels_close.append(labels[aux])
            #for the wider context
            for i in where_toponym:
                res=[]
                if i-big_left < 0:
                    res.append(text_tokenization[:i])
                    tokens_left = big_left-len(list(chain.from_iterable(res)))
                    if i+len_toponym+tokens_left+big_right > len_text:
                        res.append(text_tokenization[i:])
                    else:
                        res.append(text_tokenization[i:i+tokens_left+big_right+len_toponym])
                elif i+len_toponym+big_right > len_text:
                    tokens_left = big_right-len(text_tokenization[i:])+len_toponym
                    if i-tokens_left-big_left < 0:
                        res.append(text_tokenization[:i])
                    else:
                        res.append(text_tokenization[i-big_left-tokens_left:i])
                    res.append(text_tokenization[i:])
                else:
                    res.append(text_tokenization[i-big_left:i])
                    res.append(text_tokenization[i:i+len_toponym+big_right])

                local_wider_context.append(list(chain.from_iterable(res)))
                local_toponym_mention.append(toponym_tokenization)

            for close in local_close_context:
                close_context_test = tokenizer.encode(close, add_special_tokens=True)
                local_input_ids_close.append(close_context_test)
            for wider in local_wider_context:
                wider_context_test = tokenizer.encode(wider, add_special_tokens=True)
                local_input_ids_wider.append(wider_context_test)
            for toponym in local_toponym_mention:
                toponym_test = tokenizer.encode(toponym, add_special_tokens=True)
                local_input_ids_toponyms.append(toponym_test)

            local_input_ids_toponyms = pad_sequences(local_input_ids_toponyms, maxlen=32, dtype="long",
                         value=0, truncating="post", padding="post")
            local_input_ids_close = pad_sequences(local_input_ids_close, maxlen=64, dtype="long",
                         value=0, truncating="post", padding="post")
            local_input_ids_wider = pad_sequences(local_input_ids_wider, maxlen=MAX_LEN, dtype="long",
                         value=0, truncating="post", padding="post")

            for close in local_input_ids_close:
                input_ids_close.append(close)
            for wider in local_input_ids_wider:
                input_ids_wider.append(wider)
            for toponym in local_input_ids_toponyms:
                input_ids_toponyms.append(toponym)

            for sent in local_input_ids_wider:
                att_mask = [int(token_id > 0) for token_id in sent]
                attention_wider.append(att_mask)
            for sent in local_input_ids_close:
                att_mask = [int(token_id > 0) for token_id in sent]
                attention_close.append(att_mask)
            for sent in local_input_ids_toponyms:
                att_mask = [int(token_id > 0) for token_id in sent]
                attention_toponyms.append(att_mask)

            aux+=1
        return [[input_ids_toponyms, attention_toponyms, labels_close, land_coverage_close, elevation_close, vegetation_close], [input_ids_close, attention_close, labels_close, land_coverage_close, elevation_close, vegetation_close], [input_ids_wider, attention_wider, labels_close, land_coverage_close, elevation_close, vegetation_close]]

#Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 2
LR = 3e-5

if __name__ == "__main__":

    encoder = LabelEncoder()
    resolution = math.pow(4,4)
    region_codes = []
    #Get the Coordinates from the train file and the test file
    Y1_train,Y1_test = getCoordinatesFromTrainAndTest(input_trainfile_path, input_testfile_path)
    #Number of Train Instances
    train_instances_sz = len(Y1_train)
    #Number of Test Instances
    test_instances_sz = len(Y1_test)
    #region_codes is the region assigned to each coordinate in both train and test files
    if geoproperties == 1:
        if input_data == 0:
            land_coverage_file = rasterio.open("/content/drive/MyDrive/geoproperties/historic_landcover_1850.tif")
        else:
            land_coverage_file = rasterio.open("/content/drive/MyDrive/geoproperties/landcov_world.tif")
        elevation_file = rasterio.open("/content/drive/MyDrive/geoproperties/elevation_world.tif")
        vegetation_file = rasterio.open("/content/drive/MyDrive/geoproperties/veg_world.tif")
        lc_data = []
        el_data = []
        ve_data = []
        
    for coordinates in (Y1_train+Y1_test):
        region = latlon2healpix(coordinates[0], coordinates[1], resolution)
        region_codes.append(region)
        if geoproperties == 1:
            boundaries = healpy.boundaries(int(resolution), region)
            
            minlat = [val[0] for val in boundaries]
            minlong = [val[1] for val in boundaries]
            maxlat = [val[2] for val in boundaries]
            maxlong = [val[3] for val in boundaries]
            coords_boundaries = getLatLong([minlat, minlong, maxlat, maxlong], int(resolution))
            
            lats = [val[0] for val in coords_boundaries]
            longs = [val[1] for val in coords_boundaries]
            minx, miny = min(longs), min(lats)
            maxx, maxy = max(longs), max(lats)
            bbox = box(minx, miny, maxx, maxy)

            [lc, [el, ve]] = get_values(land_coverage_file, elevation_file, vegetation_file, bbox)     
            lc_data.append(lc)
            el_data.append([el])
            ve_data.append([ve])

    if geoproperties == 1:
        Y3_train = lc_data[:train_instances_sz]
        Y3_test = lc_data[-test_instances_sz:]
        Y3_train = np.array(Y3_train)
        Y3_test = np.array(Y3_test)
        print("Y3_train shape:",Y3_train.shape)
        print("Y3_test shape:",Y3_test.shape)

        Y4_train = el_data[:train_instances_sz]
        Y4_test = el_data[-test_instances_sz:]
        Y4_train = np.array(Y4_train)
        Y4_test = np.array(Y4_test)
        print("Y4_train shape:",Y4_train.shape)
        print("Y4_test shape:",Y4_test.shape)

        Y5_train = ve_data[:train_instances_sz]
        Y5_test = ve_data[-test_instances_sz:]
        Y5_train = np.array(Y5_train)
        Y5_test = np.array(Y5_test)
        print("Y5_train shape:",Y5_train.shape)
        print("Y5_test shape:",Y5_test.shape)

    #turn the regions into labels
    classes = to_categorical(encoder.fit_transform(region_codes))
    Y1_train = np.array(Y1_train)
    Y1_test = np.array(Y1_test)
    #labels associated with the train file
    Y2_train = classes[:train_instances_sz]
    #labels associated with the test file
    Y2_test = classes[-test_instances_sz:]
    
    region_list = [i for i in range(Y2_train.shape[1])]
    region_classes = encoder.inverse_transform(region_list)

    codes_matrix = []
    for i in range(len(region_classes)):
        [xs, ys, zs] = healpy.pix2vec( int(resolution), region_classes[i] )
        codes_matrix.append([xs, ys, zs])
    #codes_matrix is the matrix that associates each label with the centroid coordinates of its region
    num_labels = len(classes[0])
  
    
    if geoproperties == 1:
        lc_matrix = []
        el_matrix = []
        ve_matrix = []
        for region in region_classes:
            boundaries = healpy.boundaries(int(resolution), region)
            
            minlat = [val[0] for val in boundaries]
            minlong = [val[1] for val in boundaries]
            maxlat = [val[2] for val in boundaries]
            maxlong = [val[3] for val in boundaries]
            coords_boundaries = getLatLong([minlat, minlong, maxlat, maxlong], int(resolution))
            
            lats = [val[0] for val in coords_boundaries]
            longs = [val[1] for val in coords_boundaries]
            minx, miny = min(longs), min(lats)
            maxx, maxy = max(longs), max(lats)
            bbox = box(minx, miny, maxx, maxy)

            [lc, [el, ve]] = get_values(land_coverage_file, elevation_file, vegetation_file, bbox)     
            lc_matrix.append(lc)
            el_matrix.append([el])
            ve_matrix.append([ve])
            
    #Get the BERT tokens for each input
    #Different approaches were used depending on the train model
    train_toponym, train_close_context, train_wider_context = getBERTInputs(input_trainfile_path, Y2_train, Y3_train, Y4_train, Y5_train)
    test_toponym, test_close_context, test_wider_context = getBERTInputs(input_testfile_path, Y2_test, Y3_test, Y4_test, Y5_test)
    
    del Y2_train
    gc.collect()
    del Y2_test
    gc.collect()
    del Y3_train
    gc.collect()
    del Y3_test
    gc.collect()
    del Y4_train
    gc.collect()
    del Y4_test
    gc.collect()
    del Y5_train
    gc.collect()
    del Y5_test
    gc.collect()
    
    #Split coordinates into train & validation
    if input_data == 3:
        _, _, training_coords, valid_coords = train_test_split(train_toponym[0],Y1_train[:LINES], random_state=seed_val, test_size=0.001 )
        split_size = 0.001
    else:
        _, _, training_coords, valid_coords = train_test_split(train_toponym[0],Y1_train[:LINES], random_state=seed_val, test_size=0.1 )
        split_size = 0.1
    testing_coords = Y1_test[:LINES]

    #Split training into train & validation
    train_toponym, validation_toponym = trainingValidationSplit(train_toponym, split_size)
    train_close_context, validation_close_context = trainingValidationSplit(train_close_context, split_size)
    train_wider_context, validation_wider_context = trainingValidationSplit(train_wider_context, split_size)
    
    #Convert to PyTorch data types
    #Toponyms

    train_toponym_inputids = torch.tensor(train_toponym[0])
    train_toponym_attentionmasks = torch.tensor(train_toponym[1])
    train_toponym_labels = torch.tensor(train_toponym[2])
    train_toponym_land_coverage = torch.tensor(train_toponym[3])
    train_toponym_elevation = torch.tensor(train_toponym[4])
    train_toponym_vegetation = torch.tensor(train_toponym[5])

    validation_toponym_inputids = torch.tensor(validation_toponym[0])
    validation_toponym_attentionmasks = torch.tensor(validation_toponym[1])
    validation_toponym_labels = torch.tensor(validation_toponym[2])
    validation_toponym_land_coverage = torch.tensor(validation_toponym[3])
    validation_toponym_elevation = torch.tensor(validation_toponym[4])
    validation_toponym_vegetation = torch.tensor(validation_toponym[5])

    train_toponym_data = TensorDataset(train_toponym_inputids, train_toponym_attentionmasks, train_toponym_labels, train_toponym_land_coverage, train_toponym_elevation, train_toponym_vegetation)
    train_toponym_dataloader = DataLoader(train_toponym_data, batch_size=BATCH_SIZE)
    del train_toponym_inputids
    gc.collect()
    del train_toponym_attentionmasks
    gc.collect()
    del train_toponym_labels
    gc.collect()
    del train_toponym_land_coverage
    gc.collect()
    del train_toponym_elevation
    gc.collect()
    del train_toponym_vegetation
    gc.collect()
    del train_toponym_data
    gc.collect()

    validation_toponym_data = TensorDataset(validation_toponym_inputids, validation_toponym_attentionmasks, validation_toponym_labels, validation_toponym_land_coverage, validation_toponym_elevation, validation_toponym_vegetation)
    validation_toponym_dataloader = DataLoader(validation_toponym_data, batch_size=BATCH_SIZE)
    del validation_toponym_inputids
    gc.collect()
    del validation_toponym_attentionmasks
    gc.collect()
    del validation_toponym_labels
    gc.collect()
    del validation_toponym_land_coverage
    gc.collect()
    del validation_toponym_elevation
    gc.collect()
    del validation_toponym_vegetation
    gc.collect()
    del validation_toponym_data
    gc.collect()

    #Close Context

    train_close_inputids = torch.tensor(train_close_context[0])
    train_close_attentionmasks = torch.tensor(train_close_context[1])

    validation_close_inputids = torch.tensor(validation_close_context[0])
    validation_close_attentionmasks = torch.tensor(validation_close_context[1])

    train_close_data = TensorDataset(train_close_inputids, train_close_attentionmasks)
    train_close_dataloader = DataLoader(train_close_data, batch_size=BATCH_SIZE)

    del train_close_inputids
    gc.collect()
    del train_close_attentionmasks
    gc.collect()
    del train_close_data
    gc.collect()

    validation_close_data = TensorDataset(validation_close_inputids, validation_close_attentionmasks)
    validation_close_dataloader = DataLoader(validation_close_data, batch_size=BATCH_SIZE)

    del validation_close_inputids
    gc.collect()
    del validation_close_attentionmasks
    gc.collect()
    del validation_close_data
    gc.collect()

    #Wide Context

    train_wide_inputids = torch.tensor(train_wider_context[0])
    train_wide_attentionmasks = torch.tensor(train_wider_context[1])

    validation_wide_inputids = torch.tensor(validation_wider_context[0])
    validation_wide_attentionmasks = torch.tensor(validation_wider_context[1])

    train_wide_data = TensorDataset(train_wide_inputids, train_wide_attentionmasks)
    train_wide_dataloader = DataLoader(train_wide_data, batch_size=BATCH_SIZE)

    del train_wide_inputids
    gc.collect()
    del train_wide_attentionmasks
    gc.collect()
    del train_wide_data
    gc.collect()

    validation_wide_data = TensorDataset(validation_wide_inputids, validation_wide_attentionmasks)
    validation_wide_dataloader = DataLoader(validation_wide_data, batch_size=BATCH_SIZE)

    del validation_wide_inputids
    gc.collect()
    del validation_wide_attentionmasks
    gc.collect()
    del validation_wide_data
    gc.collect()
            
    test_toponym_inputids = torch.tensor(test_toponym[0])
    test_toponym_attentionmasks = torch.tensor(test_toponym[1])
    test_toponym_labels = torch.tensor(test_toponym[2])

    test_toponym_data = TensorDataset(test_toponym_inputids, test_toponym_attentionmasks, test_toponym_labels)
    test_toponym_dataloader = DataLoader(test_toponym_data, batch_size=BATCH_SIZE)

    del test_toponym_inputids
    gc.collect()
    del test_toponym_attentionmasks
    gc.collect()
    del test_toponym_labels
    gc.collect()
    del test_toponym_data
    gc.collect()


    test_close_inputids = torch.tensor(test_close_context[0])
    test_close_attentionmasks = torch.tensor(test_close_context[1])


    test_close_data = TensorDataset(test_close_inputids, test_close_attentionmasks)
    test_close_dataloader = DataLoader(test_close_data, batch_size=BATCH_SIZE)

    del test_close_inputids
    gc.collect()
    del test_close_attentionmasks
    gc.collect()
    del test_close_data
    gc.collect()


    test_wide_inputids = torch.tensor(test_wider_context[0])
    test_wide_attentionmasks = torch.tensor(test_wider_context[1])

    test_wide_data = TensorDataset(test_wide_inputids, test_wide_attentionmasks)
    test_wide_dataloader = DataLoader(test_wide_data, batch_size=BATCH_SIZE)

    del test_wide_inputids
    gc.collect()
    del test_wide_attentionmasks
    gc.collect()
    del test_wide_data
    gc.collect()

        

    #Coordinates

    train_coords = torch.tensor(training_coords)
    validation_coords = torch.tensor(valid_coords)
    
    train_coords_data = TensorDataset(train_coords)
    train_coords_dataloader = DataLoader(train_coords_data, batch_size=BATCH_SIZE)

    validation_coords_data = TensorDataset(validation_coords)
    validation_coords_dataloader = DataLoader(validation_coords_data, batch_size=BATCH_SIZE)

    del train_coords 
    gc.collect()
    del validation_coords
    gc.collect()
    del train_coords_data 
    gc.collect()
    del validation_coords_data
    gc.collect()
    
    test_coords = torch.tensor(testing_coords)

    test_coords_data = TensorDataset(test_coords)
    test_coords_dataloader = DataLoader(test_coords_data, batch_size=BATCH_SIZE)

    del test_coords
    gc.collect()
    del test_coords_data
    gc.collect()
    
    torch.cuda.empty_cache()
    #TRAINING THE MODEL
    
    n_train_steps = int(len(train_toponym[0]) / BATCH_SIZE * EPOCHS)
    model = LanguageModel(num_labels)
    #model.load_state_dict(torch.load("/content/drive/MyDrive/TR/corpora/SpatialML/bert-geo-spatialml.pt"))
    optimizer = AdamW (model.parameters(), lr=LR)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=n_train_steps
    )
    print("")
    print("Number of classes -> {}".format(num_labels))
    print("")
    print("Number of Training instances -> {}".format(len(train_toponym[0])))
    print("")
    print("Number of Validation instances -> {}".format(len(validation_toponym[0])))
    
    codes_matrix = torch.tensor(codes_matrix)
    land_coverage_matrix = torch.tensor(lc_matrix)
    elevation_matrix = torch.tensor(el_matrix)
    vegetation_matrix = torch.tensor(ve_matrix)

    for epoch in range(EPOCHS):
        t0 = time.time()
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHS))
        print('Training...')
        loss, train_class_accuracy, training_mean_distance, train_at161_accuracy, training_median_distance = trainLoopFunction(train_toponym_dataloader, train_close_dataloader, train_wide_dataloader, train_coords_dataloader, model, optimizer, device, codes_matrix,  land_coverage_matrix, elevation_matrix, vegetation_matrix, scheduler )
        print("")
        print("  Average training loss: {0:.2f}".format(loss))
        print("")
        print("  Average training class accuracy: {0:.2f}".format(train_class_accuracy/len(train_toponym[0])))
        print("")
        print("  Mean training distance: {0:.2f}".format(training_mean_distance))
        print("")
        print("  Average training at 161km accuracy: {0:.3f}".format(train_at161_accuracy/len(train_toponym[0])))
        print("")
        print("  Median training distance: {0:.2f}".format(training_median_distance))
        print("")
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        print("")
        print("Running Validation...")
        t0 = time.time()
        val_class_accuracy, val_mean_distance, val_at161_accuracy, val_median_distance = evalLoopFunction(validation_toponym_dataloader, validation_close_dataloader,validation_wide_dataloader, validation_coords_dataloader, model, device, codes_matrix)
        print("")
        print("  Average validation class accuracy: {0:.2f}".format(val_class_accuracy/len(validation_toponym[0])))
        print("")
        print("  Mean validation distance: {0:.2f}".format(val_mean_distance))
        print("")
        print("  Average validation at 161km accuracy: {0:.3f}".format(val_at161_accuracy/len(validation_toponym[0])))
        print("")
        print("  Median validation distance: {0:.2f}".format(val_median_distance))
        print("")
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        print("")
    #torch.save(model.state_dict(), "/content/drive/MyDrive/TR/corpora/SpatialML/bert-geo-spatialml.pt")
    del train_toponym_dataloader
    gc.collect()
    del train_close_dataloader
    gc.collect()
    del train_wide_dataloader
    gc.collect()
    del train_coords_dataloader
    gc.collect()
    del validation_toponym_dataloader
    gc.collect()
    del validation_close_dataloader
    gc.collect()
    del validation_wide_dataloader
    gc.collect()
    del validation_coords_dataloader
    gc.collect()
    print("Number of Test instances -> {}".format(len(test_toponym[0])))
    print("")
    print('Running Predictions...')
    print("")
    t0 = time.time()
    test_class_accuracy, test_mean_distance, test_at161_accuracy, test_median_distance = evalLoopFunction(test_toponym_dataloader, test_close_dataloader,test_wide_dataloader, test_coords_dataloader, model, device, codes_matrix)
    print("  Average test class accuracy: {0:.2f}".format(test_class_accuracy/len(test_toponym[0])))
    print("")
    print("  Mean test distance: {0:.2f}".format(test_mean_distance))
    print("")
    print("  Average test at 161km accuracy: {0:.3f}".format(test_at161_accuracy/len(test_toponym[0])))
    print("")
    print("  Median test distance: {0:.2f}".format(test_median_distance))
    print("")
    print("  Test took: {:}".format(format_time(time.time() - t0)))
