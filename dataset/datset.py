from torch.utils.data.dataset import Dataset
import librosa
from utils.helper_code import *

class ECG_Dataset(Dataset) :
    def __init__(self,header_files, recording_files, leads, sample_length, sample_rate = 500) :
        super(Dataset,self).__init__()
        self.header_files = header_files
        self.recording_files = recording_files
        self.leads = leads
        self.sample_length = sample_length    
        self.sample_rate = sample_rate
    def __len__(self) :
         
        return len(self.recording_files)
    
    def __getitem__(self,index) :
        header = load_header(self.header_files[index])
        orig_sr = int(header.split(' ')[2])
        #print(orig_sr)
        recording = load_recording(self.recording_files[index])
        recordings = choose_leads(recording, header, self.leads)
        data = np.zeros((recordings.shape[0],self.sample_length))
        for i in range(len(recordings)):
            #print(type(data[i]))
            y = librosa.resample(recordings[i].astype(np.float), orig_sr, self.sample_rate, res_type='kaiser_best') 
            #print(y.shape[0])
            if y.shape[0] < self.sample_length :
                
                data[i,0:y.shape[0]] = y
            elif y.shape[0] >= self.sample_length:
                data[i] = y[0:self.sample_length]
        current_labels = get_labels(header)
        #print(current_labels)
        labels = np.zeros(( num_classes))
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[j] = 1
#         data =recordings[:,0:self.sample_length]
       #data = data*10/np.linalg.norm(data)
#         for i in range(len(data)):
#             data[i] = data[i]/max(abs(data[i]))
#         orig_sr = int(header.split(' ')[2])
#         for i in range(len(data)):
#             #print(type(data[i]))
#             data[i] = librosa.resample(data[i].astype(np.float), orig_sr, self.sample_rate, res_type='kaiser_best')        
        recording_id = get_recording_id(header)
        if data.shape[1] < self.sample_length :
            print(data)
        return (data,labels,self.header_files[index])