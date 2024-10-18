# 0.まえおき
[フォーク元はこちら](https://github.com/yxlllc/contentvec)  
[公式はこちら]([https://github.com/yxlllc/contentvec](https://github.com/auspicious3000/contentvec) 
本リポジトリはフォーク元のcontentvec学習の日本語版です。  
Linux環境の想定で書いてます。windows環境で構築する際は各所読み替えてください。  

# 1.環境構築：
```
git clone https://github.com/furukawamea/contentvec.git
```
```
cd contentvec
conda create -n contentvec python=3.10
conda activate contentvec
```
```
pip install -r requirements.txt
```
```
git clone https://github.com/bfloat16/fairseq.git
git checkout 42ea630a9879121c942a7cd7b9d5e3e19e74814b

apt-get install ninja-build

cd fairseq
pip install --editable ./
python setup.py build_ext --inplace
cd ..
rsync -a contentvec/ fairseq/fairseq/

```
以下モデルをダウンロードしてください。  
[contentvec](https://ibm.ent.box.com/s/nv35hsry0v2y595etzysgnn2amsxxb0u)


# 2.前処理
```
python 00_resampler.py --in_dir [file_path] --num_gpus [gpu_count]
python 01_train_valid_tsv.py
python 02_create_contentvec_dict.py
python 03_dump_hubert_feature.py
python 04_me.py
python 05_learn_kmeans.py
python 06_dump_km_label.py
```

# 3.学習
シングルGPUでの実行例
```
chmod 755 run_pretrain_single.sh
./run_pretrain_single.sh
```

# 4.レガシーモデル変換(option)
```
./run_pretrain_single.sh
```
