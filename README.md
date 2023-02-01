A project for fine-tuning stable-diffusion models.

#How To Use It
1. Install diffusers from source
'''
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
'''
2. Clone or download repository in a different folder
'''
git clone https://github.com/rincon-santi/fine-tune-stable-diffusion.git
'''
3. Download [files](https://drive.google.com/drive/folders/1BO_dyz-p65qhBRRMRA4TbZ8qW4rB99JZ) to data/dog
4. Run dvc pipeline
'''
dvc repro
'''
Fine tuned model will be stored in trained_model
Pipeline may execute disordered (Pending fix)