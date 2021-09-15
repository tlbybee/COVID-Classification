import csv
import os
import pickle
from os import listdir
from os.path import isfile, join
import shutil

# Verify the current working directory.  Depending on your environment / editor, it may not be where you think it is
#     since paths are relative this will be problematic, so double check

cwd = os.getcwd()
print("\nCurrent working directory: {0}\n".format(cwd))

data_root_dir = '../original data/covid-chestxray-dataset-master/'
image_root_dir = '../original data/covid-chestxray-dataset-master/images'
image_tgt_dir_normal = '../../images/normal'
image_tgt_dir_covid = '../../images/covid-19'
image_tgt_dir_pneumonia = '../../images/pneumonia'
info_file_name = 'metadata.csv'
info_path = os.path.join(data_root_dir, info_file_name)

data_dict = {}

covid = ['COVID-19']
pneumonia_virus = ['SARS']
pneumonia_bacteria = ['Pneumocystis','Streptococcus']
normal = ['No Finding']
x0 = 0
x1 = 0
x2 = 0
x3 = 0
n_ct = 0
n_ards = 0

#print("\nPre CSV Loop: \n")
with open(info_path,'r') as f:

  csv_reader = csv.reader(f)
  i = 0 
  #print("\n\tPre Row Loop:")
  for row in csv_reader:
    #print("\n\t\tIn Row Loop:")    
    if i == 0:
      i += 1
      continue
    patient_id = row[0]
    subject_id = row[1]
    view = row[18]
    image_name = row[23]
    disease = row[4]
    modality = row[19]

    #print("\n\t\tmodality is : {0}\n".format(modality))

    if 'X-ray' not in modality:
      n_ct += 1
      continue
    
    #print("\n\t\t\tPre JPEG Path:")        
    jpg_path = os.path.join(image_root_dir, image_name)

    #print("\n\t\tImage Path is : {0}\n".format(jpg_path))

    if os.path.exists(jpg_path) and ('AP' in view or 'PA' in view):    
      if data_dict.get(patient_id+'_'+subject_id) is None:
          data_dict[patient_id+'_'+subject_id] = {'class':{
                                              'COVID-19':0,
                                              'pneumonia_virus':0,
                                              'pneumonia_bacteria':0,
                                              'normal':0
                                              },
                                              'image_dict':{}}
      if disease == 'ARDS':
        n_ards += 1
        continue
      #CCB 04/09/21 - Bad IF statements since disease can be a / delimited list
      # which would never match an item in the source lists as a full string ....
      # need to split disease by / and search each/every element against the source
      # list 
      #if disease in covid:

      disease = disease.split('/')[-1]

      if disease in covid:
        #print("\n\tFound COVID Image")              
        data_dict[patient_id+'_'+subject_id]['class']['COVID-19'] = 1
        x0 += 1
        #Copy the image to the covid folder 
        shutil.copyfile(jpg_path, os.path.join(image_tgt_dir_covid, image_name))

      if disease in pneumonia_virus:
        data_dict[patient_id+'_'+subject_id]['class']['pneumonia_virus'] = 1
        x1 += 1
        #Copy the image to the pneumonia folder 
        if ('AP' in view):         
          shutil.copyfile(jpg_path, os.path.join(image_tgt_dir_pneumonia, image_name))

      if disease in pneumonia_bacteria:
        data_dict[patient_id+'_'+subject_id]['class']['pneumonia_bacteria'] = 1
        x2 += 1
        #Copy the image to the pneumonia folder 
        if ('AP' in view):                 
          shutil.copyfile(jpg_path, os.path.join(image_tgt_dir_pneumonia, image_name))
      if disease in normal:
        data_dict[patient_id+'_'+subject_id]['class']['normal'] = 1
        x3 += 1
        #Copy the image to the normal folder 
        if ('AP' in view):                 
          shutil.copyfile(jpg_path, os.path.join(image_tgt_dir_normal, image_name))

      print("\n\tPatient ID is : : {0}\n".format(patient_id))          
      print("\n\t\tView is : : {0}\n".format(view))        
      data_dict[patient_id+'_'+subject_id]['image_dict'][image_name] = {
        'path':jpg_path, 
        'type':view
      }


y0 = 0
y1 = 0
y2 = 0
y3 = 0
z0 = 0
z1 = 0
z2 = 0
z3 = 0
v0 = 0
v1 = 0
v2 = 0
v3 = 0
w0 = 0
w1 = 0
w2 = 0
w3 = 0
i = 0
j = 0
ap_list = []
pa_list = []
for key, value in data_dict.items():
  for jpg_name, jpg_info in value['image_dict'].items():
    print(jpg_info['type'])
    y0 += value['class']['COVID-19']
    y1 += value['class']['pneumonia_virus']
    y2 += value['class']['pneumonia_bacteria']
    y3 += value['class']['normal']
    j += 1
    if 'PA' in jpg_info['type'] or 'AP' in jpg_info['type']:
      i += 1
      z0 += value['class']['COVID-19']
      z1 += value['class']['pneumonia_virus']
      z2 += value['class']['pneumonia_bacteria']
      z3 += value['class']['normal']
      if 'PA' in jpg_info['type']:
        print("\n\tIn PA loop")                      
        pa_list.append(jpg_name)
        v0 += value['class']['COVID-19']
        v1 += value['class']['pneumonia_virus']
        v2 += value['class']['pneumonia_bacteria']
        v3 += value['class']['normal']
      if 'AP' in jpg_info['type']:
        #print("\n\tIn AP loop")                      
        ap_list.append(jpg_name)
        w0 += value['class']['COVID-19']
        w1 += value['class']['pneumonia_virus']
        w2 += value['class']['pneumonia_bacteria']
        w3 += value['class']['normal']
                 
print (x0, x1, x2, x3)
print (i, j)
print (y0, y1, y2, y3)
print (z0, z1, z2, z3)
print (v0, v1, v2, v3)
print (w0, w1, w2, w3)

pickle.dump(data_dict, open('../data_preprocess/formal_covid_dict_ap.pkl','wb'))
pickle.dump(ap_list, open('ap_list.pkl','wb'))
pickle.dump(pa_list, open('pa_list.pkl','wb'))
saved_path = '../data_preprocess/formal_covid_dict.pkl'
if os.path.exists(saved_path):
  os.remove(saved_path)
pickle.dump(data_dict, open(saved_path,'wb'))
print ('finish')
##
###             Xray	
