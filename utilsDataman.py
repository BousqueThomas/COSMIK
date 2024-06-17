"""Manages the movement and use of data files."""

import os
import warnings
from scipy.spatial.transform import Rotation as R

import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd

class TRCFile(object):
    """A plain-text file format for storing motion capture marker trajectories.
    TRC stands for Track Row Column.

    The metadata for the file is stored in attributes of this object.

    See
    http://simtk-confluence.stanford.edu:8080/display/OpenSim/Marker+(.trc)+Files
    for more information.

    """
    def __init__(self, fpath=None, **kwargs):
            #path=None,
            #data_rate=None,
            #camera_rate=None,
            #num_frames=None,
            #num_markers=None,
            #units=None,
            #orig_data_rate=None,
            #orig_data_start_frame=None,
            #orig_num_frames=None,
            #marker_names=None,
            #time=None,
            #):
        """
        Parameters
        ----------
        fpath : str
            Valid file path to a TRC (.trc) file.

        """
        self.marker_names = []
        if fpath != None:
            self.read_from_file(fpath)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)

    # def read_from_file(self, fpath):
    #     # Read the header lines / metadata.
    #     # ---------------------------------
    #     # Split by any whitespace.
    #     # TODO may cause issues with paths that have spaces in them.
    #     f = open(fpath)
    #     # These are lists of each entry on the first few lines.
    #     first_line = f.readline().split()
    #     # Skip the 2nd line.
    #     f.readline()
    #     third_line = f.readline().split()
    #     fourth_line = f.readline().split()
    #     f.close()

    #     # First line.
    #     if len(first_line) > 3:
    #         self.path = first_line[3]
    #     else:
    #         self.path = ''

    #     # Third line.
    #     self.data_rate = float(third_line[0])
    #     self.camera_rate = float(third_line[1])
    #     self.num_frames = int(third_line[2])
    #     self.num_markers = int(third_line[3])
    #     self.units = third_line[4]
    #     self.orig_data_rate = float(third_line[5])
    #     self.orig_data_start_frame = int(third_line[6])
    #     self.orig_num_frames = int(third_line[7])

    #     # Marker names.
    #     # The first and second column names are 'Frame#' and 'Time'.
    #     self.marker_names = fourth_line[2:]

    #     len_marker_names = len(self.marker_names)
    #     if len_marker_names != self.num_markers:
    #         warnings.warn('Header entry NumMarkers, %i, does not '
    #                 'match actual number of markers, %i. Changing '
    #                 'NumMarkers to match actual number.' % (
    #                     self.num_markers, len_marker_names))
    #         self.num_markers = len_marker_names

    #     # Load the actual data.
    #     # ---------------------
    #     col_names = ['frame_num', 'time']
    #     # This naming convention comes from OpenSim's Inverse Kinematics tool,
    #     # when it writes model marker locations.
    #     for mark in self.marker_names:
    #         col_names += [mark + '_tx', mark + '_ty', mark + '_tz']
    #     dtype = {'names': col_names,
    #             'formats': ['int'] + ['float64'] * (3 * self.num_markers + 1)}
    #     usecols = [i for i in range(3 * self.num_markers + 1 + 1)]
    #     self.data = np.loadtxt(fpath, delimiter='\t', skiprows=5, dtype=dtype,
    #                            usecols=usecols)
    #     self.time = self.data['time']

    #     # Check the number of rows.
    #     n_rows = self.time.shape[0]
    #     if n_rows != self.num_frames:
    #         warnings.warn('%s: Header entry NumFrames, %i, does not '
    #                 'match actual number of frames, %i, Changing '
    #                 'NumFrames to match actual number.' % (fpath,
    #                     self.num_frames, n_rows))
    #         self.num_frames = n_rows



    def read_from_file(self, fpath):
        # Read the header lines / metadata.
        # ---------------------------------
        # Split by any whitespace.
        # TODO may cause issues with paths that have spaces in them.
        with open(fpath, 'r') as file:
            # These are lists of each entry on the first few lines.
            first_line = file.readline().strip().split(',')

            # Determine the number of markers from the length of the first line.
            self.num_markers = (len(first_line) -2) // 3  #on n'a pas le temps et le numero de la frame donc il faut enlever le -2
            
            # # Skip the 2nd line.
            # file.readline()
            # third_line = file.readline().strip().split(',')
            # fourth_line = file.readline().strip().split(',')
            # print('lenfisrtline', len(first_line))
            # print('firstline =',first_line)
            # print('thirdline =',third_line)
            # print('fourthline =',fourth_line)
            file.close()

            
        trc_data = pd.read_csv(fpath, header=None)
        # print('trc_data shape=' , trc_data.shape[0])
    
        num_frames = trc_data.shape[0]

        # # First line.
        # if len(first_line) > 3:
        #     self.path = first_line[3]
        #     print('first_line[3] =', first_line[3])
        # else:
        #     self.path = ''

        self.path='/home/tbousquet/Documents/Challenge'
        # print('self.path =', self.path)

        # Third line.
        self.data_rate = float(60.0)
        self.camera_rate = float(60.0)
        self.num_frames = int(num_frames)
        self.num_markers = int(20)
        self.units = 'm'
        self.orig_data_rate = float(60.0)
        self.orig_data_start_frame = int(1)
        self.orig_num_frames = int(111)

        # print('type data rate=', type(self.data_rate), type(self.camera_rate),type(self.num_frames),type(self.num_markers),type(self.units),type(self.orig_data_rate),type(self.orig_data_start_frame), type(self.orig_num_frames))



        # Marker names.
        self.marker_names = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']
        # print(type(self.marker_names),print(self.marker_names))


        # Load the actual data.
        # ---------------------
        col_names = ['frame_num', 'time']
        # This naming convention comes from OpenSim's Inverse Kinematics tool,
        # when it writes model marker locations.
        for mark in self.marker_names:
            col_names += [mark + '_tx', mark + '_ty', mark + '_tz']
            # print('col-name', len(col_names))
        # print('long col name =' ,len(col_names))





        dtype = {'names': col_names,'formats': ['int']+ ['float64'] * (3 * self.num_markers+1)} #Il faut enlever le +1 car la colone temps n'existe pas dans notre cas => creation d'un dictionnaire
        usecols = [i for i in range(3 * self.num_markers +1+1)]
        # print('usecols=', usecols)
        self.data = np.loadtxt(fpath, delimiter=',', skiprows=0, dtype=dtype)
        # print('self.data est longue de :' ,len(self.data),self.data)
        self.time = self.data['time']
        # print('self.time est :', self.time)

        # # Check the number of rows.
        # n_rows = len(self.data)
        # print('n_rows =', n_rows)
        # if n_rows != self.num_frames:
        #     warnings.warn('%s: Header entry NumFrames, %i, does not '
        #                 'match actual number of frames, %i, Changing '
        #                 'NumFrames to match actual number.' % (fpath,self.num_frames, n_rows))
        









    def __getitem__(self, key):
        """See `marker()`.

        """
        return self.marker(key)
    
    def units(self):
        return self.units




    # def time(self):
    #     return self.data['time']

    def time(self):
        this_dat = np.empty((self.num_frames, 1))
        this_dat[:, 0] = self.time
        print('this_dat de la classe : ', this_dat)
        return this_dat        

    def marker(self, name):
        """The trajectory of marker `name`, given as a `self.num_frames` x 3
        array. The order of the columns is x, y, z.

        """
        this_dat = np.empty((self.num_frames, 3))
        this_dat[:, 0] = self.data[name + '_tx']
        this_dat[:, 1] = self.data[name + '_ty']
        this_dat[:, 2] = self.data[name + '_tz']
        return this_dat

    def add_marker(self, name, x, y, z):
        """Add a marker, with name `name` to the TRCFile.

        Parameters
        ----------
        name : str
            Name of the marker; e.g., 'R.Hip'.
        x, y, z: array_like
            Coordinates of the marker trajectory. All 3 must have the same
            length.

        """
        if (len(x) != self.num_frames or len(y) != self.num_frames or len(z) !=
                self.num_frames):
            raise Exception('Length of data (%i, %i, %i) is not '
                    'NumFrames (%i).', len(x), len(y), len(z), self.num_frames)
        self.marker_names += [name]
        self.num_markers += 1
        if not hasattr(self, 'data'):
            self.data = np.array(x, dtype=[('%s_tx' % name, 'float64')])
            self.data = append_fields(self.data,
                    ['%s_t%s' % (name, s) for s in 'yz'],
                    [y, z], usemask=False)
        else:
            self.data = append_fields(self.data,
                    ['%s_t%s' % (name, s) for s in 'xyz'],
                    [x, y, z], usemask=False)

    def marker_at(self, name, time):
        x = np.interp(time, self.time, self.data[name + '_tx'])
        y = np.interp(time, self.time, self.data[name + '_ty'])
        z = np.interp(time, self.time, self.data[name + '_tz'])
        return [x, y, z]

    def marker_exists(self, name):
        """
        Returns
        -------
        exists : bool
            Is the marker in the TRCFile?

        """
        return name in self.marker_names



    def write(self, fpath):
        """Write this TRCFile object to a TRC file.

        Parameters
        ----------
        fpath : str
            Valid file path to which this TRCFile is saved.

        """
        f = open(fpath, 'w')

        # Line 1.
        f.write('PathFileType  4\t(X/Y/Z) %s\n' % os.path.split(fpath)[0])

        # Line 2.
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
                'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')

        # Line 3.
        f.write('%.1f\t%.1f\t%i\t%i\t%s\t%.1f\t%i\t%i\n' % (
            self.data_rate, self.camera_rate, self.num_frames,
            self.num_markers, self.units, self.orig_data_rate,
            self.orig_data_start_frame, self.orig_num_frames))

        # Line 4.
        f.write('Frame#\tTime\t')
        for imark in range(self.num_markers):
            f.write('%s\t\t\t' % self.marker_names[imark])
        f.write('\n')

        # Line 5.
        f.write('\t\t')
        for imark in np.arange(self.num_markers) + 1:
            f.write('X%i\tY%s\tZ%s\t' % (imark, imark, imark))
        f.write('\n')

        # Line 6.
        f.write('\n')

        # Data.
        for iframe in range(self.num_frames):
            f.write('%i' % (iframe + 1))
            f.write('\t%.7f' % self.time[iframe])
            for mark in self.marker_names:
                idxs = [mark + '_tx', mark + '_ty', mark + '_tz']
                f.write('\t%.7f\t%.7f\t%.7f' % tuple(
                    self.data[coln][iframe] for coln in idxs))
            f.write('\n')

        f.close()

    def add_noise(self, noise_width):
        """ add random noise to each component of the marker trajectory
            The noise mean will be zero, with the noise_width being the
            standard deviation.

            noise_width : int
        """
        for imarker in range(self.num_markers):
            components = ['_tx', '_ty', '_tz']
            for iComponent in range(3):
                # generate noise
                noise = np.random.normal(0, noise_width, self.num_frames)
                # add noise to each component of marker data.
                self.data[self.marker_names[imarker] + components[iComponent]] += noise
                
    def rotate(self, axis, value):
        """ rotate the data.

            axis : rotation axis
            value : angle in degree
        """
        for imarker in range(self.num_markers):   
            
            temp = np.zeros((self.num_frames, 3))
            temp[:,0] = self.data[self.marker_names[imarker] + '_tx']
            temp[:,1] = self.data[self.marker_names[imarker] + '_ty']
            temp[:,2] = self.data[self.marker_names[imarker] + '_tz']          
            
            r = R.from_euler(axis, value, degrees=True)            
            temp_rot = r.apply(temp)
            
            self.data[self.marker_names[imarker] + '_tx'] = temp_rot[:,0]
            self.data[self.marker_names[imarker] + '_ty'] = temp_rot[:,1]
            self.data[self.marker_names[imarker] + '_tz'] = temp_rot[:,2]
            
    def offset(self, axis, value):
        """ offset the data.

            axis : rotation axis
            value : offset in m
        """
        for imarker in range(self.num_markers):            
            if axis.lower() == 'x':
                self.data[self.marker_names[imarker] + '_tx'] += value
            elif axis.lower() == 'y':
                self.data[self.marker_names[imarker] + '_ty'] += value       
            elif axis.lower() == 'z':
                self.data[self.marker_names[imarker] + '_tz'] += value          
            else:
                raise ValueError("Axis not recognized")
