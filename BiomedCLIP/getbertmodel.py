from open_clip import create_model_from_pretrained, get_tokenizer, add_model_config
import torch

pretrained = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/code/BiomedCLIP/open_clip_pytorch_model.bin'
open_clip_config = '/ailab/public/pjlab-smarthealth03/leiwenhui/thr/code/BiomedCLIP/open_clip_config.json'
tokenizer_config = 'tokenizer_config.json'
add_model_config(open_clip_config)
model, preprocess = create_model_from_pretrained(model_name='open_clip_config', pretrained=pretrained)
tokenizer = get_tokenizer(model_name='open_clip_config')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"running on {device}")
model.to(device)
model.eval()

context_length = 256

bert = model.text
input_ids = 'this is a photo of Synthetic Brain Tumor MRI'
input_ids = tokenizer(input_ids,context_length=context_length).to(device)
print(input_ids)
print('\n')
outputs = bert(input_ids)
# 获取最后一层的隐藏状态作为文本编码
print(outputs)
