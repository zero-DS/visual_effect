# make_metadata.py
diffusers의 lora 훈련 스크립트는 폴더 내에 학습할 이미지들과, 그 이미지들의 metadata를 담은 metadata.jsonl 파일을 포함해야 합니다.
학습할 이미지 파일명을 객체의 종류를 나타내도록 저장하였기 때문에, metadata를 만들 때 이미지 파일명을 활용할 수 있습니다. lora 학습시 배울 concept/style을 특정 keyword로 배우도록 s3wnf3lt와 같은 keyword와 함께 해당 이미지를 표현하는 간단한 prompt를 metadata에 저장합니다.

# lora.sh
lora를 학습시켜 safetensors 파일을 얻습니다. 쉘파일에서 실행하는 파이썬 스크립트의 --num_train_epochs 및 --rank를 조절하며 학습시킬 수 있습니다.

# run.sh
위에서 학습시킨 lora 파일을 사용해서 원본 이미지의 질감을 바꿉니다. 쉘파일에서 실행하는 파이썬 스크립트의 --use_controlnet, --strength, --guidance_scale 등을 조절하여 생성할 수 있습니다. --input_path는 질감을 바꿀 이미지들이 있는 **폴더**의 경로, --lora_path는 lora safetensors **파일**의 경로이며 이 두 argument는 필수로 명시해줘야합니다.
