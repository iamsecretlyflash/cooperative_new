### Installation
```
#conda create -n cooperative python=3.9 && conda activate cooperative
pip install -e .
cd translation && pip install -e .
pip install -r requirements.txt
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install evaluate
pip install huggingface-hub -U
```

### Run GLUE Tasks
```
sh ./scripts/run_glue.sh
```