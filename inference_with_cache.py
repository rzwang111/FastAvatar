import numpy as np
import cv2, os, argparse, audio
import subprocess
from tqdm import tqdm
import torch, face_detection
from models import Wav2Lip,split_Wav2Lip
# from testsrc import run_gpu_monitoring_and_training
# from testsrc import *
from my_cache import MyCache, RowItem
import platform
import time
# import librosa
import sys
import os
import sys
# import librosa
# import librosa.display

import numpy as np

import torch
import torchvision as tv

# import matplotlib.pyplot as plt
  
from PIL import Image
import pandas as pd

torch.set_grad_enabled(False)


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=128)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=64)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

# parser.add_argument('--ws', default=10, type=int,
# 					help='size of slide window')

parser.add_argument('--cs', default=70, type=int,
					help='Cache size')

parser.add_argument('--sr', default=0.9, type=float,
					help='skip rate')

parser.add_argument('--strategy', default='nearest', type=str,
					help='replace strategy')
parser.add_argument('--tmp', default='./results/my_cache/result.avi', type=str,
					help='temp video')

args = parser.parse_args()
args.img_size = 96

args.wav2lip_batch_size = 1

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True
	
def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def select(now_ae,database_aes):
	ans_max_sim = 1.0
	ans = None
	for db_ae in database_aes:
		now_sim = cosine_sims(db_ae,now_ae)
		if now_sim > ans_max_sim:
			ans = db_ae
			ans_max_sim = now_sim
	return ans 
def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)
	batch_size = args.face_det_batch_size
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def cut_audio(audio_data, start_time, end_time, sample_rate=16000):
    # 计算开始和结束的样本点
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # 使用NumPy切片操作进行切割
    cut_audio_data = audio_data[start_sample:end_sample]
    
    return
def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path,modelx):
	model = modelx()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '').replace('audio_encoder.', 'audio_encoder_blocks.')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def mainbefore():
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'
	mel = 0
	wav = audio.load_wav(args.audio, 16000)   # (采样率，)     16000   /1s 
	# wav1 = audio.load_wav(args.audio, 16000)   # (采样率，)     16000   /1s 
	mel = audio.melspectrogram(wav)
	return mel,fps,full_frames,wav

def mel_to_wav(mel, sr=16000, n_fft=2048, hop_length=512, ref_level_db=20):
    
    # 1. 从 dB 转换回幅度谱
    mel = mel + ref_level_db
    mel = np.exp(mel / 20.0)
    
    # 2. 将 Mel 频谱转换回 STFT 频谱
    stft_matrix = librosa.feature.inverse.mel_to_stft(mel, sr=sr, n_fft=n_fft)
    
    # 3. 使用 Griffin-Lim 算法恢复音频
    track = librosa.griffinlim(stft_matrix, hop_length=hop_length)
    
    return track

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def print_mean_and_variance(times, name, width=15 ):
    if len(times) > 0:
        mean = np.mean(times)
        std_dev = np.std(times)
        print(f"{name:<{width}} - Mean: {mean:.4e} seconds, Standard Deviation: {std_dev:.4e}")
    else:
        print(f"{name:<{width}} - No data available")


skiprate = args.sr
cache_size = args.cs
strategy  = args.strategy
FPS = 100
if __name__ == '__main__':
	
	t_frame = []
	t_face_encoder = []
	t_audio_encoder =[]
	t_query = []
	t_set_threshold = []
	t_decoder = []
	t_cache_update = []

	# mycache = MyCache(cache_size, N)
	mycache = MyCache(size=cache_size,strategy=strategy)

	mel,fps,full_frames,wav = mainbefore()
	sp_wav_model = load_model(args.checkpoint_path,split_Wav2Lip)

	outdir = './results/my_cache/'
	os.makedirs(outdir,exist_ok=True)
	# args.outfile = './results/my_cache/testCache.mp4'.format()
	temp_video = args.tmp

	sumhit=0
	usehit=0

	print ("Number of frames available for inference: "+str(len(full_frames)))
	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	wav_l = []
	i = 0

	while 1:
		start_idx = int(i * mel_idx_multiplier)
		start_time = float(i) / float(fps)
		
		
		if start_idx + mel_step_size > len(mel[0]):
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])   # num，80，16
		i += 1

	print(len(mel[0]))
	model = sp_wav_model
	model.eval()

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]
	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)   
	

	print ("Model loaded")


	preds= []

	try:
		model = model.to(device)
	except:
		pass

	pre_audio = 0
	# embeddings = []
	thresholds = []
	errors = []
	sr_l = []
	all_frames = []
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		# print(coords)
		# break
		t_start = time.time()
		if i == 0:
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(temp_video, 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
			
		face_sequences = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		audio_sequences = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		input_dim_size = len(face_sequences.size())


		if input_dim_size > 4:
			audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0) 
			face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
		with torch.no_grad():
			flag = True

			#  Todo 
			#  time of face encoder

			if i == 0:
				t1 = time.time()
				feats = sp_wav_model.face_encoder(face_sequences)
				t2 = time.time()
				t_face_encoder.append(t2-t1)

			t1 = time.time()
			audio_embedding = sp_wav_model.audio_encoder(audio_sequences)
			t2 = time.time()
			t_ae = t2-t1
			t_audio_encoder.append(t2-t1)
			f = feats
			sumhit+=1
			# # # # ####
			# # if query

			
			new_input = audio_embedding.reshape(-1)
			# embeddings.append(new_input.cpu())

			# sample_l.append(RowItem(new_input, None))


			t1 = time.time()
			pred = mycache.query(new_input)

			# sim_l.append(max_sim)

			t2 = time.time()

			t_query.append(t2-t1)

			#  time of query 

			if pred is None:
				t1 = time.time()			
				pred = sp_wav_model.face_decoder(feats,audio_embedding,input_dim_size,1)
				pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255
				t2 = time.time()
				t_d = t2-t1 
				t_decoder.append(t2-t1)

				t1 = time.time()
				mycache.update(new_input, pred)
				t2 = time.time()
				t_cache_update.append(t2-t1)
				all_frames.append(1)
				
			else:
				all_frames.append(0)
				usehit+=1
			# skiprate_change = 1 - (1./FPS - t_ae)/t_d
			# if (skiprate_change - skiprate)**2 > 0.0025:
			# 	skiprate = skiprate_change
			# 	usehit=0 
			# 	sumhit=1e-6
			pred = pred
			sr = usehit/sumhit
			# if mycache.is_full() and (len(sample_l) % N == 0):
			# if mycache.is_full():
			t1 = time.time()
			# sr_l.append(sr)
			# mycache.set_threshold(skip_rate=skiprate, sr=sr)
			# thresholds.append(mycache.smoothed_threshold)
			if i>0:
				t_avg = np.mean(t_frame)  # if len(t_frame)<500 else np.mean(t_frame[-500:])
				fps_ = 1/t_avg
				mycache.set_threshold(skip_rate=FPS, sr=fps_)
				thresholds.append(mycache.smoothed_threshold)
				errors.append(fps_)
			# sample_l = []
			# sim_l = []
			t2 = time.time()
			t_set_threshold.append(t2-t1)

			######

			######

			# # not query
			# # time of decoder
			

			# t1 = time.time()			
			# pred = sp_wav_model.face_decoder(feats,audio_embedding,input_dim_size,1)
			# pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255
			# t2 = time.time()
			# t_decoder.append(t2-t1)
			# ######

			preds.append(pred)
			t_end = time.time()
			t_frame.append(t_end - t_start)

			

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
			f[y1:y2, x1:x2] = p
			out.write(f)
	print_mean_and_variance(t_face_encoder, "Face Encoder")
	print_mean_and_variance(t_audio_encoder, "Audio Encoder")
	print_mean_and_variance(t_decoder, "Face Decoder")
	print_mean_and_variance(t_frame, "One Frame")

	print_mean_and_variance(t_query, "Query")
	print_mean_and_variance(t_set_threshold, "Set Threshold")
	print_mean_and_variance(t_cache_update, "Cache Update")
	# Create a DataFrame from the collected data
	data = {
		'Frame Time': t_frame,
		'All Frames': all_frames
	}

	df = pd.DataFrame(data)

	# Save the DataFrame to an Excel file
	output_file = f'output_data_{skiprate}.xlsx'
	df.to_excel(output_file, index=False)

	print(f"Data saved to {output_file}")
	out.release()
	# import matplotlib.pyplot as plt

	# # # Plot the thresholds and skiprate
	# plt.figure(figsize=(10, 6))
	# plt.plot(sr_l, label='Skip Rate')
	# # plt.axhline(y=skiprate, color='r', linestyle='--', label='Skip Rate')
	# plt.xlim(50,7000)
	# plt.xlabel('Iterations')
	# plt.ylabel('Value')
	# plt.title('Skip Rate Over Time')
	# plt.legend()
	# plt.savefig(f'./pics/skiprate.png')
	# plt.close()
	# plt.figure(figsize=(10, 6))
	# plt.plot(thresholds, label='Threshold')
	# plt.xlim(50,7000)
	# plt.xlabel('Iterations')
	# plt.ylabel('Value')
	# plt.title('Threshold Over Time')
	# plt.legend()
	# plt.savefig(f'./pics/threshold.png')
	# plt.close()	
	# plt.figure(figsize=(10, 6))
	# plt.plot(errors, label='Error')
	# plt.xlim(50,7000)
	# plt.xlabel('Iterations')
	# plt.ylabel('Value')
	# plt.title('Error Over Time')
	# plt.legend()
	# plt.savefig(f'./pics/error.png')
	# plt.close()
	# plt.show()
	# # # print(sr_l[0])
	

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, temp_video, args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')
	# print(usehit)
	realsr = usehit/sumhit
	FPS = f'{1/np.mean(t_frame):.2f}'
	print("skiprate:{}, cache_size:{}, sr:{}, FPS:{}, usehit:{}, sumhit:{} ".format(skiprate, cache_size, realsr, FPS, usehit, sumhit))
	
	
	# import matplotlib.pyplot as plt
	# from sklearn.decomposition import PCA
	# import numpy as np
	# embeddings_matrix = np.vstack(embeddings[:100])
	# pca = PCA(n_components=2)

	# # 执行 PCA 降维
	# reduced_embeddings = pca.fit_transform(embeddings_matrix)

	# # 可视化
	# fig = plt.figure(figsize=(8, 6))
	# # plt = fig.add_subplot(111, projection='2d')

	# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue', label='Reduced Data')
	# plt.title('PCA Reduced Data Visualization (2D)')
	# plt.xlabel('Principal Component 1')
	# plt.ylabel('Principal Component 2')
	# # plt.set_zlabel('Principal Component 3')
	# plt.legend()

	# plt.savefig('2d-100.png')

# python inference_with_cache.py --checkpoint_path ./checkpoints/wav2lip.pth --audio ./WDA_AdamSchiff_000.wav --face ./0.jpg
