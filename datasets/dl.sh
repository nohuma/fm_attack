export HF_ENDPOINT=https://hf-mirror.com
hf download lerobot/pi0_libero_finetuned_v044 --local-dir /root/autodl-tmp/pi0_libero_finetuned

hf download lerobot/pi05_libero_finetuned_v044 --local-dir /root/autodl-tmp/pi05_libero_finetuned

hf download hf://lerobot/pi05_libero_finetuned_v044/model.safetensors

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

hf download google/paligemma-3b-pt-224 --local-dir /root/autodl-tmp/paligemma-3b-pt-224

hf download openvla/openvla-7b-finetuned-libero-spatial --local-dir /root/autodl-tmp/openvla-7b-finetuned-libero-spatial
hf download lerobot/pi0_libero_base --local-dir /root/autodl-tmp/pi0_libero_base
