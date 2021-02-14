import pydicom as dicom
from copy import deepcopy
import numpy as np
import pickle
import os
import sys
import getopt
import logging
import time
import functools
import patient_v2

def get_logger(name):
    """
    Returns an already configured logger for a specific module.
    (This should be used instead of stdout.)
    :param name: the name of the modeule where the logger is created
    :return: a custom configured logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("hypertrophy.log", mode='a')
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s -- %(msg)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


class DCMreaderVM:

    def __init__(self, folder_name):
        '''
        Reads in the dcm files in a folder which corresponds to a patient.
        It follows carefully the physical slice locations and the frames in a hearth cycle.
        It does not matter if the location is getting higher or lower. 
        '''
        self.num_slices = 0
        self.num_frames = 0
        self.broken = False
        images = []
        slice_locations = []
        file_paths = []        

        dcm_files = sorted(os.listdir(folder_name))
        dcm_files = [d for d in dcm_files if len(d.split('.')[-2]) < 4]
        if len(dcm_files) == 0:  # sometimes the order number is missing at the end
            dcm_files = sorted(os.listdir(folder_name))

        for file in dcm_files:

            if file.find('.dcm') != -1:
                try:
                    temp_ds = dicom.dcmread(os.path.join(folder_name, file))
                    images.append(temp_ds.pixel_array)
                    slice_locations.append(temp_ds.SliceLocation)
                    file_paths.append(os.path.join(folder_name, file))
                except:
                    self.broken = True
                    return
        
        current_sl = -1
        frames = 0
        increasing = False
        indices = []
        for idx, slice_loc in enumerate(slice_locations):
            if abs(slice_loc - current_sl) > 0.01:  # this means a new slice is started
                self.num_slices += 1
                self.num_frames = max(self.num_frames, frames)
                frames = 0
                indices.append(idx)

                if (slice_loc - current_sl) > 0.01:
                    increasing = True
                else:
                    increasing = False
                
                current_sl = slice_loc
            frames += 1

        if self.num_slices != 0 and self.num_frames != 0:
            self.load_matrices(images, indices, increasing, slice_locations, file_paths)
        else:
            logger.warning("There are no frames. This folder should be deleted. Path: {}".format(folder_name))
        self.num_images = len(images)

    def load_matrices(self, images, indices, increasing, slice_locations, file_paths):
        size_h, size_w = images[0].shape
        self.dcm_images = np.ones((self.num_slices, self.num_frames, size_h, size_w))
        self.dcm_slicelocations = np.ones((self.num_slices, self.num_frames, 1))
        self.dcm_file_paths = np.zeros((self.num_slices, self.num_frames), dtype=object)

        for i in range(len(indices) - 1):

            for idx in range(indices[i], indices[i + 1]):
                slice_idx = (i if increasing else (len(indices) - 1 - i))
                frame_idx = idx - indices[i]
                if images[idx].shape == self.dcm_images[slice_idx, frame_idx, :, :].shape:
                    self.dcm_images[slice_idx, frame_idx, :, :] = images[idx]
                    self.dcm_slicelocations[slice_idx, frame_idx, 0] = slice_locations[idx]
                    self.dcm_file_paths[slice_idx, frame_idx] = file_paths[idx]
                else:
                    logger.error('Wrong shape at {}'.format(file_paths[idx]))

        for idx in range(indices[-1], len(images)):
            slice_idx = (len(indices) - 1 if increasing else 0)
            frame_idx = idx - indices[-1]
            if self.dcm_images.shape[1] == frame_idx:
                logger.info(file_paths[idx])
            if images[idx].shape == self.dcm_images[slice_idx, frame_idx, :, :].shape:
                self.dcm_images[slice_idx, frame_idx, :, :] = images[idx]
                self.dcm_slicelocations[slice_idx, frame_idx, 0] = slice_locations[idx]
                self.dcm_file_paths[slice_idx, frame_idx] = file_paths[idx]
            else:
                logger.error('Wrong shape at {}'.format(file_paths[idx]))

    def get_image(self, slice, frame):
        return self.dcm_images[slice, frame, :, :]
    
    def get_slicelocation(self, slice, frame):
        return self.dcm_slicelocations[slice, frame, 0]

    def get_dcm_path(self,slice, frame):
        return self.dcm_file_paths[slice, frame]


class CONreaderVM:

    def __init__(self, file_name):
        """
        Reads in a con file and saves the curves grouped according to its corresponding slice, frame and place.
        Finds the tags necessary to calculate the volume metrics.
        """
        self.file_name = file_name
        self.container = []
        self.contours = None

        con_tag = "XYCONTOUR"  # start of the contour data
        stop_tag = "POINT"     # if this is available, prevents from reading unnecessary lines
        volumerelated_tags = [
            'Study_id=',
            'Field_of_view=',
            'Image_resolution=',
            'Slicethickness=',
            'Patient_weight=',
            'Patient_height',
            'Study_description=',
            'Patient_gender='
        ]

        self.volume_data = {
            volumerelated_tags[0]: None, 
            volumerelated_tags[1]: None, 
            volumerelated_tags[2]: None,
            volumerelated_tags[3]: None,
            volumerelated_tags[4]: None,
            volumerelated_tags[5]: None,
            volumerelated_tags[6]: None,
            volumerelated_tags[7]: None
        }

        con = open(file_name, 'rt')
        
        def find_volumerelated_tags(line):
            for tag in volumerelated_tags:
                if line.find(tag) != -1:
                    value = line.split(tag)[1]  # the place of the tag will be an empty string, second part: value
                    self.volume_data[tag] = value
        
        def mode2colornames(mode):
            if mode == 0:
                return 'ln'   # left (endo)
            elif mode == 1:
                return 'lp'   # left (epi) contains the myocardium
            elif mode == 2:
                return 'rp'   # right (epi)
            elif mode == 5:
                return 'rn'   # right (endo)
            else:
                logger.warning('Unknown mode {}'.format(mode))
                return 'other'

        def find_xycontour_tag():
            line = con.readline()
            find_volumerelated_tags(line)
            while line.find(con_tag) == -1 and line.find(stop_tag) == -1 and line != "":
                line = con.readline()
                find_volumerelated_tags(line)
            return line

        def identify_slice_frame_mode():
            line = con.readline()
            splitted = line.split(' ')
            return int(splitted[0]), int(splitted[1]), mode2colornames(int(splitted[2]))

        def number_of_contour_points():
            line = con.readline()
            return int(line)

        def read_contour_points(num):
            contour = []
            for _ in range(num):
                line = con.readline()
                xs, ys = line.split(' ')
                contour.append((float(xs), float(ys)))  # unfortubately x and y are interchanged
            return contour

        line = find_xycontour_tag()
        while line.find(stop_tag) == -1 and line != "":

            slice, frame, mode = identify_slice_frame_mode()
            num = number_of_contour_points()
            contour = read_contour_points(num)
            self.container.append((slice, frame, mode, contour))
            line = find_xycontour_tag()

        con.close()
        return

    def get_hierarchical_contours(self):
        # if it is not initializedyet, then create it
        if self.contours is None:

            self.contours = {}
            for item in self.container:
                slice = item[0]
                frame = item[1]   # frame in a hearth cycle
                mode = item[2]    # mode can be red, green, yellow
                contour = item[3]

                # rearrange the contour
                d = {'x': [], 'y': []}
                for point in contour:
                    d['x'].append(point[0])
                    d['y'].append(point[1])

                if not(slice in self.contours):
                    self.contours[slice] = {}

                if not(frame in self.contours[slice]):
                    self.contours[slice][frame] = {}

                if not(mode in self.contours[slice][frame]):
                    x = d['x']
                    y = d['y']
                    N = len(x)
                    contour_mtx = np.zeros((N, 2))
                    contour_mtx[:, 0] = np.array(x)
                    contour_mtx[:, 1] = np.array(y)
                    self.contours[slice][frame][mode] = contour_mtx

        return self.contours

    def contour_iterator(self, deep=True):
        self.get_hierarchical_contours()
        for slice, frame_level in self.contours.items():
            for frame, mode_level in frame_level.items():
                if deep:
                    mode_level_cp = deepcopy(mode_level)
                else:
                    mode_level_cp = mode_level
                yield slice, frame, mode_level_cp

    def get_volume_data(self):
        # process field of view
        fw_string = self.volume_data['Field_of_view=']
        sizexsize_mm = fw_string.split('x')  # variable name shows the format
        size_h = float(sizexsize_mm[0])
        size_w = float(sizexsize_mm[1].split(' mm')[0])  # I cut the _mm ending

        # process image resolution
        img_res_string = self.volume_data['Image_resolution=']
        sizexsize = img_res_string.split('x')
        res_h = float(sizexsize[0])
        res_w = float(sizexsize[1])

        # process slice thickness
        width_string = self.volume_data['Slicethickness=']
        width_mm = width_string.split(' mm')
        width = float(width_mm[0])

        # process weight
        weight_string = self.volume_data['Patient_weight=']
        weight_kg = weight_string.split(' kg')
        weight = float(weight_kg[0])

        # process height
        # Unfortunately, patient height is not always available.
        # Study description can help in that case but its form changes heavily.
        if 'Patient_height=' in self.volume_data.keys():  
            height_string = self.volume_data['Patient_height=']
            height = height_string.split(" ")[0]
        else:
            height_string = str(self.volume_data['Study_description='])
            height = ''
            for char in height_string:
                if char.isdigit():
                    height += char
        if height == '':
            logger.warning('Unknown height in con file {}'.format(self.file_name))
            height = 178
        else:
            try:
                height = float(height)
            except ValueError:
                height = 178
                logger.error(' Wrong height format in con file {}'.format(self.file_name))

        # gender
        gender = self.volume_data['Patient_gender=']
        
        return (size_h/res_h, size_w/res_w), width, weight, height, gender

class DataLoader:
    def __init__(self, inputdir, outputdir):
        self.fileLocation = inputdir
        self.outputdir = outputdir

    def normalize(self, img):
        imin = img.min()
        imax = img.max()

        a = (255) / (imax - imin)
        b = 255 - a * imax
        new_img = (a * img + b).astype(np.uint8)
        return np.array(new_img, dtype=np.uint8)


    def sort_cons(self, directory):
        cr = CONreaderVM(directory + '/sa/contours.con')
        dr = DCMreaderVM(directory + '/sa/images')
        _, __, weight, height, gender = cr.get_volume_data()

        num = []
        slice_num = dr.num_slices
        frm_num = dr.num_frames

        gap = slice_num//3

        num.append(gap//2 + 1)

        for k in range(2):
            num.append(num[k] + gap)

        with open(directory+"/meta.txt", "r") as meta:
            pathology = meta.readline().split(' ')[1]

        images = []

        for j in range(frm_num):
            if(j % 2 == 0):
                slices = []

                for i in num:
                    slices.append(self.normalize(dr.get_image(i, j)))
                
                images.append(np.stack(slices, axis=-1))

        return pathology, weight, height, gender, np.array(images, dtype='uint8')

    def picklePatient(self, directory, id):
        pathology, weight, height, gender, images = self.sort_cons(directory)

        patient = Patient(pathology, gender, weight, height, images)

        output = self.outputdir + '/' + str(directory.split('/')[-1])
        with open(output, 'wb') as outfile:
            pickle.dump(patient, outfile)

    def readAllData(self):
        rootdir = self.fileLocation

        directories = next(os.walk(rootdir))[1]
        for i, directory in enumerate(directories):
            if not os.path.exists(self.outputdir + '/' + str(directory.split('/')[-1])):
                self.picklePatient(rootdir + directory, i)

    def unpicklePatients(self, directory):
        patients = []
        deleted = []
        for root, dirs, files in os.walk(directory):
            for i,f in enumerate(files):
                with open(directory + '/' + f, 'rb') as infile:
                    
                    pat = RenameUnpickler.renamed_load(infile)
                    if hasattr(pat, 'images'):
                        if pat.pathology in ['U18_m', 'U18_f', 'adult_m_sport', 'adult_f_sport']:
                            deleted.append(f)
                        elif len(pat.images) != 13 and len(pat.images) != 25:
                            deleted.append(f)
                        else:
                            patients.append((f,pat))
                    else:
                        deleted.append(f)
            break

        print(len(deleted))

        final = []
        for i, pat in enumerate(patients):
            if len(pat[1].images) == 25:
                final.append(i)
                temp = []
                for img in pat[1].images:
                    if not np.all(img == img[0][0]):
                        temp.append(img)
                pat[1].images = temp

        print(final)
        print(len(patients))

        result = []
        for i, pat in enumerate(patients):
            flag = True
            for img in pat[1].images:
                max_val = np.amax(img)
                if max_val == 0 or np.isnan(max_val):
                    print(pat[0])
                    flag = False
                    break
            if flag:
                result.append(pat)

        return result

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "patient":
            renamed_module = "patient_v2"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


    def renamed_load(file_obj):
        return RenameUnpickler(file_obj).load()


    def renamed_loads(pickled_bytes):
        file_obj = io.BytesIO(pickled_bytes)
        return renamed_load(file_obj)

def main(argv):
    dl = DataLoader(argv[0], argv[1])
    dl.readAllData()


if __name__ == "__main__":
    main(sys.argv[1:])
