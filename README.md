# 1.環境構築：
```
git clone https://github.com/furukawamea/contentvec.git

cd contentvec
conda create -n contentvec python=3.10
conda activate contentvec

pip install -r requirements.txt

git clone https://github.com/bfloat16/fairseq.git
git checkout 42ea630a9879121c942a7cd7b9d5e3e19e74814b

apt-get install ninja-build

cd fairseq
pip install --editable ./
python setup.py build_ext --inplace

cd ..
rsync -a contentvec/ fairseq/fairseq/



```
https://ibm.ent.box.com/s/nv35hsry0v2y595etzysgnn2amsxxb0u

https://huggingface.co/TencentGameMate/chinese-hubert-large/blob/main/chinese-hubert-large-fairseq-ckpt.pt

下载链接里面的模型，放入当前文件夹。

legacy系列不能用，砍掉了东西了。

先试试预测cnhubertlarge的13层，炸了再说。

# 2.前処理
完全轮椅，可以不分说话人，一股脑扔进去处理就行了

所有脚本里面合理修改线程数（一般一张卡给一个线程）

第5步的percent根据内存修改，不炸内存就行
```
python 00_resampler.py
python 01_train_valid_tsv.py
python 02_create_contentvec_dict.py
python 03_dump_hubert_feature.py
python 04_me.py
python 05_learn_kmeans.py
python 06_dump_km_label.py
```

# 3.学習

依据卡数修改run_pretrain_single.sh里面的distributed_training.nprocs_per_node=8
```
./run_pretrain_single.sh
```
