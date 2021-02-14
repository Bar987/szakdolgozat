import pickle
from patient_v2 import Patient
from data_loader_v2 import DataLoader
from sklearn.model_selection import train_test_split
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
DATA_PATH = 'drive/data/pickle/'

def loadPatients(dirpath, num_of_classes, seq_len, cropped = ''):
    data = []

    print(num_of_classes, seq_len,  cropped)

    dl = DataLoader('x', 'y')

    all_patients = dl.unpicklePatients(dirpath)

    sizes = []

    patients = []
    if cropped == 'cropped':
        for patient in all_patients:
            patient = list(patient)
            sizes.append(len(patient[1].images[0][0]))
            patient[1], is_valid = crop_on_contour(patient[1])

            if is_valid:
                patients.append(tuple(patient))
    else:
        patients = all_patients

    np.random.shuffle(patients)

    print(countListEntries(sizes))

    samples = []
    norm = []
    hcm = []
    other = []

    if num_of_classes == 2:
        diagnosys = {'Normal': 0, 'Other': 1}
        for patient in patients:
            if patient[1].pathology == 'Normal':
                norm.append(patient)
            else:
                other.append(patient)
    elif num_of_classes == 3:
        diagnosys = {'Normal': 0, 'HCM': 1, 'Other': 2}
        for patient in patients:
            if patient[1].pathology == 'Normal':
                norm.append(patient)
            elif patient[1].pathology =='HCM' :
                hcm.append(patient)
            else:
                other.append(patient)

    samples.append(norm)
    samples.append(hcm)
    samples.append(other)

    train = [] 
    val = []
    test = []
    for arr in samples:
        if len(arr) > 0:
            temp_train, temp_val = train_test_split(
                arr, test_size=0.2, random_state=0, shuffle=True)

            temp_val, temp_test = train_test_split(
                temp_val, test_size=0.5, random_state=0, shuffle=True)

            train += temp_train
            val += temp_val
            test += temp_test

    all_pathologies = countListPatientEntries(patients)
    train_pat = countListPatientEntries(train)
    val_pat = countListPatientEntries(val)
    test_pat = countListPatientEntries(test)
    
    print(len(patients), all_pathologies)
    print(len(train),train_pat)
    print(len(val),val_pat)
    print(len(test),test_pat)

    data = []
    for patient in train:
        if patient[1].pathology in diagnosys:
            label = diagnosys[patient[1].pathology]
        else:
            label =  diagnosys['Other']

        for i in range(len(patient[1].images) - seq_len + 1):
            img = patient[1].images[i : i + seq_len]

            data.append({'img': img, 'label': label, 'filename':patient[0]})
    with open(DATA_PATH+cropped+'train'+str(num_of_classes)+'_'+str(seq_len), 'wb') as pik:
        pickle.dump(data, pik)


    data = []
    for patient in val:
        if patient[1].pathology in diagnosys:
            label = diagnosys[patient[1].pathology]
        else:
            label =  diagnosys['Other']


        for i in range(len(patient[1].images) - seq_len + 1):
            img = patient[1].images[i : i + seq_len]

            data.append({'img': img, 'label': label, 'filename':patient[0]})

    with open(DATA_PATH+cropped+'val_'+str(num_of_classes)+'_'+str(seq_len), 'wb') as pik:
        pickle.dump(data, pik)



    data = []
    for patient in test:
        if patient[1].pathology in diagnosys:
            label = diagnosys[patient[1].pathology]
        else:
            label =  diagnosys['Other']

        for i in range(len(patient[1].images) - seq_len + 1):
            img = patient[1].images[i : i + seq_len]

            data.append({'img': img, 'label': label, 'filename':patient[0]})

    with open(DATA_PATH+cropped+'test_'+str(num_of_classes)+'_'+str(seq_len), 'wb') as pik:
        pickle.dump(data, pik)

def countListPatientEntries(arr):
    temp = []
    for i in range(len(arr)):
        temp.append(arr[i][1].pathology)
    
    count_map = dict()
    for i in temp:
        count_map[i] = count_map.get(i, 0) + 1
    
    return count_map

def countListEntries(arr):
    count_map = dict()
    for i in arr:
        count_map[i] = count_map.get(i, 0) + 1
    
    return count_map

def crop_on_contour(patient):
    max_x = 0.0
    max_y = 0.0
    min_x = 300.0
    min_y = 300.0
    is_ln = False
    max_area = 0
    great_con = ()
    for slc in patient.contours:
        for frame in patient.contours[slc]:
            if 'lp' in patient.contours[slc][frame].keys():
                if calc_area(patient.contours[slc][frame]['lp']) > max_area:
                    great_con = (slc, frame, 'lp')
                for point in patient.contours[slc][frame]['lp']:
                    if point[0] > max_x:
                        max_x = point[0]
                    if point[0] < min_x:
                        min_x = point[0]
                    if point[1] > max_y:
                        max_y = point[1]
                    if point[1] < min_y:
                        min_y = point[1]
                    
            elif 'ln' in patient.contours[slc][frame].keys():
                is_ln = True
                if calc_area(patient.contours[slc][frame]['ln']) > max_area:
                    great_con = (slc, frame, 'ln')
                for point in patient.contours[slc][frame]['ln']:
                    if point[0] > max_x:
                        max_x = point[0]
                    if point[0] < min_x:
                        min_x = point[0]
                    if point[1] > max_y:
                        max_y = point[1]
                    if point[1] < min_y:
                        min_y = point[1]
    
    # print("Endocardium ", is_ln)
    contour = patient.contours[great_con[0]][great_con[1]][great_con[2]]
    

    width = max_x - min_x
    height = max_y-min_y

    if width > height:
        max_y = max_y + (width-height)/2.0
        min_y = min_y - (width-height)/2.0
    elif height > width:
        max_x = max_x + (height-width)/2.0
        min_x = min_x - (height-width)/2.0

    width = max_x - min_x
    height = max_y-min_y

    if is_ln:
        new_width = width * 2.2
        new_height = height * 2.2
    else:
        new_width = width * 2.0
        new_height = height * 2.0

    
    max_x = max_x + (new_width - width) / 2 
    min_x = min_x - (new_width - width) / 2 
    max_y = max_y + (new_width - width) / 2 
    min_y = min_y - (new_width - width) / 2 

    max_x = int(max_x)
    max_y = int(max_y)
    min_x = int(min_x)
    min_y = int(min_y)

    arr = []
    for img in patient.images:
        if img.shape[1] > new_width and img.shape[0] > new_height:
            temp = img[min_y : max_y, min_x : max_x, :]
            arr.append(temp)
        else:
            arr.append(img)

    # print(arr[0].shape)
    
    patient.images = arr
    return patient, True

def calc_area(con):
	area = 0
	for i in range(len(con)-1):
		area += np.cross(con[i], con[i+1])
	area += np.cross(con[-1], con[0])
	return area

def draw_images(images):
    fig=plt.figure(figsize=(16, 16))
    for i in range(len(images)):
        fig.add_subplot(4, 4, i+1)
        draw(images[i])
    
    plt.show()


def draw(image):
    img_mtx = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(float)
    p1, p99 = np.percentile(img_mtx, (1, 99))
    if np.isnan(p1) or np.isnan(p99) or p99 == 0:
        p1 = 0.0
        p99 = 255.00
    img_mtx[img_mtx < p1] = p1
    img_mtx[img_mtx > p99] = p99
    img_mtx = (img_mtx - p1) / (p99 - p1)
    plt.imshow(img_mtx)

def main(argv):
    loadPatients(argv[0], 3, 13, 'cropped')
    loadPatients(argv[0], 3, 10, 'cropped')
    loadPatients(argv[0], 2, 13, 'cropped')
    loadPatients(argv[0], 3, 13)
    loadPatients(argv[0], 2, 13)
    

if __name__ == "__main__":
    main(sys.argv[1:])