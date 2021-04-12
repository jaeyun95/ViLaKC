# ViLaKC   

Thanks for [UNITER](https://github.com/ChenRocks/UNITER) and [Unicoder-VL](https://github.com/microsoft/Unicoder)   


## 1. Setting up Requirements   
* [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation)(418+)   
* [Docker](https://docs.docker.com/engine/install/ubuntu/)(19.03+)
* [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart)  
```
source launch_container.sh /media/ailab/jaeyun/UNITER/txt_db /media/ailab/jaeyun/UNITER/img_db/vcr /media/ailab/jaeyun/UNITER/finetune /media/ailab/jaeyun/UNITER/pretrained /media/ailab/jaeyun/UNITER/VCR_knowledges/topK
pip install h5py
```

## 2. Download Dataset   
* Download VCR   
```
bash download.sh $YOUR_PATH
```

* Extraction Knowledge   
[click here](https://github.com/jaeyun95/KnowledgeExtraction) and extract knowledge.   
 
## 3. Pretraining  
```
horovodrun -np 2 python pretrain.py --config config/pretrain-indomain-base-8gpu.json --output_dir /src/pretrain_vilakc
```

## 4. Fine tuning
```
horovodrun -np 2 python train_vcr.py --config config/train-vcr-base-4gpu.json --output_dir /src/vcr_vilakc
```

## 5. Evaluation
```
horovodrun -np 2 python inf_vcr.py --txt_db /txt/vcr_test.db --img_db "/img/vcr_gt_val/;/img/vcr_val/" --split val --output_dir /src/pretrain_new_3tasks --checkpoint 150000 --pin_mem --fp16
```