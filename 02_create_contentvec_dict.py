import argparse
from resemblyzer import VoiceEncoder, preprocess_wav
import torch
from os.path import join, exists, dirname, basename
from tqdm import tqdm
import librosa
import pickle
import torch.multiprocessing as mp
from torchfcpe import spawn_bundled_infer_model

def extract_embedding(filepath, encoder):
    wav = preprocess_wav(filepath)
    file_embedding = encoder.embed_utterance(wav)
    embedding = torch.tensor(file_embedding)
    return embedding

def process_files(rank, filenames, root_folder, device_id, return_dict):
    torch.cuda.set_device(device_id)
    encoder = VoiceEncoder()
    fcpe = spawn_bundled_infer_model(device=device_id)
    
    def get_f0_with_fcpe(filepath):
        audio, sr = librosa.load(filepath, sr=None, mono=True)
        _audio = torch.from_numpy(audio).to(device_id).unsqueeze(0)
        f0 = fcpe(_audio, sr=sr, decoder_mode="local_argmax", threshold=0.006)
        f0 = f0.squeeze().cpu().numpy()
        f0_p = f0[f0 > 0]
        return f0_p.min(), f0_p.max(), f0_p.mean()
    
    speaker_dict = {}
    
    for filepath in tqdm(filenames, position=rank):
        # 話者名をディレクトリ名から抽出
        speaker_id = basename(dirname(filepath))
        filepath = join(root_folder, filepath)
        if not exists(filepath):
            print(f"file {filepath} doesn't exist!")
            continue

        embedding = extract_embedding(filepath, encoder=encoder)
        try:
            f0_min, f0_max, f0_mean = get_f0_with_fcpe(filepath)
        except Exception as e:
            print(f"Error: {filepath}: {e}")
            continue
        # 話者IDをキーにして埋め込みとF0情報を保存
        if speaker_id not in speaker_dict:
            speaker_dict[speaker_id] = []
        speaker_dict[speaker_id].append((embedding.numpy(), (f0_min, f0_max, f0_mean)))
    
    return_dict[rank] = speaker_dict

def parallel_process(filenames, root_folder, num_processes):
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    num_devices = torch.cuda.device_count()
    chunk_size = len(filenames) // num_processes

    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = None if i == num_processes - 1 else (i + 1) * chunk_size
        file_chunk = filenames[start:end]
        device_id = i % num_devices
        p = mp.Process(target=process_files, args=(i, file_chunk, root_folder, device_id, return_dict))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    speaker_dict = {}
    for rank, part_dict in return_dict.items():
        # 各プロセスの結果を統合
        for speaker_id, data_list in part_dict.items():
            if speaker_id not in speaker_dict:
                speaker_dict[speaker_id] = []
            speaker_dict[speaker_id].extend(data_list)
    
    return speaker_dict

def generate_list_dict_from_list(filelist_train, filelist_val, root_folder, num_processes):
    speaker_dict = {'train': {}, 'valid': {}}
    speaker_dict['valid'] = parallel_process(filelist_val, root_folder, num_processes)
    speaker_dict['train'] = parallel_process(filelist_train, root_folder, num_processes)
    return speaker_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--input_train', type=str, default="data/00_filelist/train.tsv")
    parser.add_argument('-v', '--input_val',   type=str, default="data/00_filelist/valid.tsv")
    parser.add_argument('-d', '--dataset_dir', type=str, default="dataset_raw")
    parser.add_argument('-o', '--output',      type=str, default='data/01_spk2info.dict')
    parser.add_argument('-n', '--num_process', type=int, default=5)
    args = parser.parse_args()

    # トレーニングファイルリストの読み込み
    with open(args.input_train, "r", encoding='utf-8') as file:
        data = file.readlines()[1:]
    filelist_train = [line.split("\t")[0] for line in data]

    # 検証ファイルリストの読み込み
    with open(args.input_val, "r", encoding='utf-8') as file:
        data = file.readlines()[1:]
    filelist_val = [line.split("\t")[0] for line in data]

    # speaker_list_dictの生成（トレーニングと検証データ）
    speaker_list_dict = generate_list_dict_from_list(filelist_train, filelist_val, args.dataset_dir, args.num_process)

    # 不要なリストの削除
    del filelist_train
    del filelist_val

    # speaker_list_dictをpickleで保存
    with open(args.output, 'wb') as file:
        pickle.dump(speaker_list_dict, file)
