# Runpod_Serverless_Triton_vllm_deepseek

```
git clone https://github.com/PhanCongDuy312/Runpod_Serverless_Triton_vllm_deepseek.git

cd Runpod_Serverless_Triton_vllm_deepseek

docker build -t <your_image_name> .

docker run --gpus all -p 1234:7000 -p 1235:8001 -p 1236:8002 -p 8080:8080 <your_image_name>
```