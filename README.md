# LifeMate — α-UMi với Qwen3-4B

Triển khai phương pháp của paper.pdf: global QLoRA → ba adapter Planner / Caller / Summarizer,
cùng backbone Qwen/Qwen3-4B-Instruct-2507. Rationale được sinh từ train bằng Qwen với chỉ observation trước bước hiện tại.
Đọc [đối chiếu phương pháp](docs/method.md) và [kiểm tra dữ liệu](docs/data_audit.json).

## Chạy Kaggle

Notebook: [notebooks/kaggle_glpft.ipynb](notebooks/kaggle_glpft.ipynb).
Notebook riêng tư, GPU T4, Internet bật để tải model và cài dependency; dữ liệu gắn từ cymeu4l0t/zaloai-dataset.
Notebook kiểm tra SHA256 train phải trùng bản audit, chốt revision Hugging Face và ghi commit nguồn.
Chọn PROFILE = "smoke", "pilot" hoặc "full" trong cell đầu.

Nếu Kaggle CLI đã xác thực:

~~~bash
kaggle kernels push -p notebooks
kaggle kernels status lctrnnguynkhi/lifemate-paper-qwen3-glpft
kaggle kernels output lctrnnguynkhi/lifemate-paper-qwen3-glpft -p outputs/kaggle
~~~

Nếu MCP không gắn dataset hoặc không trả trạng thái/log, dùng SDK qua cùng tài khoản đã cấu hình:

~~~bash
pip install -r requirements-kaggle.txt
python kaggle_control.py push notebooks
python kaggle_control.py status lctrnnguynkhi/lifemate-paper-qwen3-glpft
python kaggle_control.py logs lctrnnguynkhi/lifemate-paper-qwen3-glpft --seconds 30
python kaggle_control.py output lctrnnguynkhi/lifemate-paper-qwen3-glpft --output outputs/kaggle
~~~

SDK nhận credential theo cấu hình Kaggle thông thường. Nếu credential đã nằm trong một biến môi trường tên khác,
truyền tên biến qua --token-env trước tên lệnh; không truyền giá trị token trên dòng lệnh.
Notebook tìm manifest theo thư mục đóng gói, không quét toàn bộ cây audio/environment.

Artifact trong /kaggle/working/paper_run/<profile> gồm rationale, prepared data, manifest split,
adapter global, ba adapter local, lineage/tensor hashes, loss trước/sau, validation metrics, latency và submission public.
Chỉ source và báo cáo tổng hợp được đưa vào Git; không commit dữ liệu thi, rationale chứa train, model hoặc secrets.

## Chạy trực tiếp trên GPU

Dùng môi trường riêng với PyTorch 2.9.1/CUDA 12.8 để tránh thay đổi kernel theo image Kaggle:

~~~bash
python setup_training_env.py --venv-dir /tmp/lifemate_training_env
/tmp/lifemate_training_env/bin/python finetune.py --data-root ../train+public_test --output-dir outputs/full --profile full
~~~

T4 dùng FP16; BF16 chỉ bật cho GPU hỗ trợ bằng phần cứng. Loader kiểm tra prefill, padding và cached decode
trước khi tải model. Nếu image không có kernel SDPA phù hợp, dùng attention đầy đủ theo query tiles có checkpoint.

Mỗi stage chạy trong process riêng để giải phóng VRAM.
Rationale có cache; trainer hỗ trợ --resume-from-checkpoint.
Runner từ chối ghi đè adapter hoàn tất; tiếp tục các stage còn thiếu bằng lệnh module:

~~~bash
python -m umilife.train --stage caller --data-dir outputs/full/prepared \
  --output-dir outputs/full --global-adapter outputs/full/adapters/global
~~~

Truyền đúng --revision trong run_config.json khi chạy riêng stage.
Inference bằng model/adapter đã tải trên ổ đĩa:

~~~bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run.py \
  --data-root /data --split public_test --model models/qwen \
  --adapter-dir models/adapters --asr-model models/whisper --output result/submission.jsonl --fp16
~~~

Inference đọc tool qua đúng zaic_tools của split, không lấy dữ liệu app bằng cách mở environment.
Mặc định deadline 30 giây kể cả ASR. Trace giữ lý do dừng, số giây, câu trả lời và tool observations.

## Kiểm tra

~~~bash
python -B -m unittest discover -s tests -v
python audit_data.py --data-root ../train+public_test --verify-tools --output docs/data_audit.json
~~~

Unit test xác nhận loss mask, không lộ gold vào prompt, annotation prefix, chia tập theo nhóm request,
schema rejection không bị thực thi và scorer không đếm trùng call.
Kiểm tra GPU và chất lượng phải lấy từ artifact notebook; unit test hoặc replay gold không phải kết quả fine-tuning.

## Đóng gói offline

Sau khi có artifact huấn luyện:

~~~bash
python bundle_models.py --run-dir outputs/full --output models
docker build -t lifemate-paper .
docker run --rm --gpus all --network none -v /path/to/data:/data:ro \
  -v /path/to/result:/app/result lifemate-paper
~~~

Docker mặc định đọc /data/private_test/private_test.jsonl. Nếu BTC mount trực tiếp thư mục split,
cần bố trí lại đường dẫn root/split tương ứng trước khi chạy. Dockerfile là cấu hình tham khảo,
chưa build hoặc benchmark trên A30 trong phiên này.

## Tài liệu

- [Paper và điều chỉnh cho cuộc thi](docs/method.md)
- [Kết quả kiểm tra local và trạng thái Kaggle](docs/validation.md)
- [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [PEFT 0.18 API](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model)
- [Whisper small CTranslate2](https://huggingface.co/Systran/faster-whisper-small)
