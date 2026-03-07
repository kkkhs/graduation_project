# DRENet 璺ㄤ富鏈?浜戞湇鍔″櫒杩佺Щ涓庣画璺戞墜鍐岋紙Docker锛?

## 1. 鐩爣涓庡師鍒?
- 鐩爣锛氬湪 4090 鎴栦簯鏈嶅姟鍣ㄤ笂鐩存帴鎷夎捣 DRENet 璁粌锛屽苟鏀寔 checkpoint 缁窇銆?
- 鏍稿績鍘熷垯锛氫唬鐮佺増鏈浐瀹氥€侀暅鍍忕増鏈浐瀹氥€佹暟鎹缃寕杞姐€佸叏绋嬬暀鐥曘€亀andb 鍙拷婧€?
- 闀滃儚鍛藉悕瑙勮寖锛歚drenet-train:<git_sha>-cu124`

## 2. 鍓嶇疆鏉′欢
- Linux 涓绘満宸插畨瑁咃細
  - NVIDIA 椹卞姩
  - Docker
  - `nvidia-container-toolkit`
- 宸插噯澶囨暟鎹洰褰曪紙涓绘満渚э級锛?
  - `<DATASET_ROOT>/train/images|labels|degrade`
  - `<DATASET_ROOT>/val/images|labels|degrade`
  - `<DATASET_ROOT>/test/images|labels|degrade`
- 宸插噯澶?wandb 鍑瘉锛?
  - `export WANDB_API_KEY=<your_key>` 鎴栧凡 `wandb login`

## 3. 鐩綍涓庢帴鍙ｆ爣鍑?
- 瀹夸富鏈虹洰褰曪細
  - 椤圭洰锛歚<PROJECT_ROOT>`
  - 鏁版嵁锛歚<DATASET_ROOT>`
  - 杈撳嚭锛歚<RUNS_ROOT>`
  - 褰掓。锛歚<CHECKPOINT_ROOT>`
- 瀹瑰櫒鍐呭浐瀹氭槧灏勶細
  - 浠ｇ爜锛歚/workspace/project`
  - 鏁版嵁锛歚/workspace/datasets/LEVIR-Ship`
  - 杈撳嚭锛歚/workspace/runs`
  - 褰掓。锛歚/workspace/checkpoints`
- 瀹為獙鍛藉悕锛歚drenet_{dataset}_{imgsz}_{bs}_{seed}_{date}`

## 4. 蹇€熷紑濮嬶紙棣栨锛?
```bash
cd <PROJECT_ROOT>
export WANDB_API_KEY=<your_key>

bash deploy/docker/run_formal.sh \
  --mode fresh \
  --dataset-root <DATASET_ROOT> \
  --runs-root <RUNS_ROOT> \
  --checkpoints-root <CHECKPOINT_ROOT> \
  --epochs 300 \
  --batch 4 \
  --workers 4 \
  --imgsz 512 \
  --seed 42 \
  --exp-name drenet_levirship_512_bs4_s42_20260307
```

## 5. 缁窇绛栫暐锛?060 -> 4090锛?

### 5.1 涓ユ牸鍙瘮缁窇锛堥粯璁ゆ帹鑽愶級
- 鐩爣锛氬悓 commit銆佸悓鏁版嵁銆佸悓瓒呭弬锛屼粠 `last.pt` 寤剁画 epoch銆?
- 鍛戒护锛?
```bash
bash deploy/docker/run_formal.sh \
  --mode strict-resume \
  --dataset-root <DATASET_ROOT> \
  --runs-root <RUNS_ROOT> \
  --checkpoints-root <CHECKPOINT_ROOT> \
  --resume-ckpt <PATH_TO_LAST_PT> \
  --exp-name <鍘焑xp_name鎴栫画璺慹xp_name>
```
- 璇存槑锛氬缓璁敤浜庘€淒ocker 娴佺▼浜у嚭鐨?checkpoint -> Docker 娴佺▼缁窇鈥濄€?

### 5.2 鎻愰€熼噸寮€锛堝彲閫夛級
- 鐩爣锛氫粠鍚?checkpoint 鍚姩鏂?run锛屽彲鎸?4090 璧勬簮涓婅皟 batch/workers銆?
- 鍛戒护锛?
```bash
bash deploy/docker/run_formal.sh \
  --mode speed-restart \
  --dataset-root <DATASET_ROOT> \
  --runs-root <RUNS_ROOT> \
  --checkpoints-root <CHECKPOINT_ROOT> \
  --resume-ckpt <PATH_TO_BEST_OR_LAST_PT> \
  --epochs 200 \
  --batch 8 \
  --workers 8 \
  --imgsz 512 \
  --seed 42 \
  --exp-name drenet_levirship_512_bs8_s42_20260308 \
  --extra-tags resumed_from:<old_run_id>
```

## 6. wandb 瑙勮寖
- 鍥哄畾 `project=graduation-drenet`
- `run.name=<exp_name>`
- tags 鏈€灏戝寘鍚細`dataset,imgsz,bs,seed,stage=formal,host`
- 鍚屼竴 run 寮哄埗缁啓锛堜粎鍦ㄤ綘鏄庣‘闇€瑕佹椂锛夛細
```bash
export WANDB_RUN_ID=<existing_run_id>
export WANDB_RESUME_MODE=must
```

## 6.1 300 -> 1000 缁绀轰緥
- 瑙勫垯锛歚--epochs` 鏄€滄€荤洰鏍?epoch鈥濄€?- 渚嬪宸茶窇鍒?300锛屾兂缁х画鍒?1000锛?
```bash
bash deploy/docker/run_formal.sh \
  --mode speed-restart \
  --dataset-root <DATASET_ROOT> \
  --runs-root <RUNS_ROOT> \
  --checkpoints-root <CHECKPOINT_ROOT> \
  --resume-ckpt <PATH_TO_LAST_PT> \
  --epochs 1000 \
  --batch 8 \
  --workers 8 \
  --imgsz 512 \
  --seed 42 \
  --exp-name drenet_levirship_512_bs8_s42_20260308
```

- 濡傛灉瑕佸啓鍥炲悓涓€涓?wandb run锛堥潪榛樿锛夛細
```bash
export WANDB_RUN_ID=<old_run_id>
export WANDB_RESUME_MODE=must
```

## 7. 鐣欑棔涓庝骇鐗?- 鍛戒护鏃ュ織锛歚<RUNS_ROOT>/trace/train_<exp_name>_<timestamp>.log`
- 璁粌杈撳嚭锛歚<RUNS_ROOT>/<exp_name>/...`
- 褰掓。鏉冮噸锛歚<CHECKPOINT_ROOT>/drenet/<exp_name>_{best,last}.pt`
- 鏂囨。鍥炲～锛?
  - `docs/experiments/logs/exp-YYYYMMDD-xx-*.md`
  - `docs/results/baselines.md`锛堝疄楠孖D + W&B Run锛?

## 8. 楠岃瘉娓呭崟锛堣縼绉诲悗蹇呴』杩囷級
- 闀滃儚鍙敤锛氬鍣ㄥ唴 `torch.cuda.is_available()==True`
- 鏁版嵁涓€鑷达細train/val/test 鏍锋湰璁℃暟涓庡師鏈轰竴鑷?
- 缁窇鍙敤锛歚--mode strict-resume` 鎴?`--mode speed-restart` 鑷冲皯璺戦€?1-2 epoch
- 杩芥函瀹屾暣锛氳兘浠庣粨鏋滆〃鍥炴函鍒版棩蹇椼€乧heckpoint銆亀andb run

## 9. 甯歌闂
- `CUDA not available`锛氫紭鍏堟鏌?`nvidia-container-toolkit` 涓?`docker run --gpus all`銆?
- `W&B credentials missing`锛氬厛瀵煎嚭 `WANDB_API_KEY` 鎴栧畬鎴?`wandb login`銆?
- `resume` 璺緞閿欒锛氱‘璁?`--resume-ckpt` 鎸囧悜鐪熷疄 `.pt` 鏂囦欢銆?



## 10. DRENet Local Compatibility Patch
- Patch file: `patches/drenet_local_compat_20260307.patch`
- Purpose:
  - Fix PyTorch 2.6 `torch.load` compatibility (`weights_only=False` where needed)
  - Replace removed NumPy alias `np.int` with `int`
  - Fix integer clamp bound typing in `utils/loss.py`

Apply from repo root:

```bash
git apply --verbose patches/drenet_local_compat_20260307.patch
```

If direct apply fails due to context mismatch:

```bash
git apply --3way --verbose patches/drenet_local_compat_20260307.patch
```
