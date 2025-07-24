# Dự Án Bộ Định Giá Sản Phẩm (The Product Pricer)

## Tổng Quan Dự Án

Dự án này phát triển một mô hình machine learning có khả năng ước tính giá trị của sản phẩm dựa trên mô tả chi tiết của nó. Đây là một bài toán regression phức tạp trong lĩnh vực Natural Language Processing (NLP), sử dụng cả mô hình truyền thống và Large Language Models (LLMs) tiên tiến.

### Mục Tiêu
- Xây dựng hệ thống định giá sản phẩm tự động từ mô tả văn bản
- So sánh hiệu suất giữa các phương pháp machine learning khác nhau
- Triển khai fine-tuning trên mô hình ngôn ngữ lớn mã nguồn mở
- Đánh giá và tối ưu hóa hiệu suất mô hình

## Cấu Trúc Dự Án

```
nlp-llm/
├── week1/                    # Tuần 1: Xử lý dữ liệu và mô hình cơ bản
│   ├── day1.ipynb           # Khám phá và làm sạch dữ liệu
│   ├── day2.ipynb           # Mở rộng dataset và tạo features
│   ├── day3.ipynb           # Mô hình Linear Regression
│   ├── day4.ipynb           # Semantic Search và RAG
│   ├── day4-results.ipynb   # Kết quả semantic search
│   ├── day5.ipynb           # Fine-tuning GPT-4o-mini
│   ├── day5-results.ipynb   # Kết quả fine-tuning OpenAI
│   ├── items.py             # Class Item để xử lý dữ liệu
│   ├── loaders.py           # Utilities để tải dữ liệu
│   └── testing.py           # Framework đánh giá mô hình
├── week2/                    # Tuần 2: Fine-tuning mô hình mã nguồn mở
│   ├── day1.ipynb           # Giới thiệu QLoRA
│   ├── day2.ipynb           # Đánh giá mô hình gốc
│   ├── day3 and 4.ipynb     # Huấn luyện với QLoRA
│   ├── day5.ipynb           # Đánh giá mô hình đã fine-tune
│   └── *.py                 # Scripts Python tương ứng
├── requirements.txt          # Dependencies
├── environment.yml          # Conda environment
└── README.md               # Tài liệu này
```

## Chi Tiết Quy Trình Triển Khai

### TUẦN 1: XỬ LÝ DỮ LIỆU VÀ MÔ HÌNH CƠ BẢN

#### Ngày 1: Khám Phá và Làm Sạch Dữ Liệu

**Mục tiêu**: Hiểu cấu trúc dữ liệu Amazon Reviews 2023 và chuẩn bị dữ liệu cho huấn luyện.

**Nguồn dữ liệu**: 
- Dataset: McAuley-Lab/Amazon-Reviews-2023
- Focus: Đồ gia dụng (Appliances) từ Amazon
- URL: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

**Quy trình xử lý**:

1. **Tải và khám phá dữ liệu**:
   ```python
   dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                         "raw_meta_Appliances", 
                         split="full", 
                         trust_remote_code=True)
   ```

2. **Phân tích thống kê**:
   - Tổng số sản phẩm đồ gia dụng
   - Tỷ lệ sản phẩm có giá (~% có giá hợp lệ)
   - Phân phối độ dài mô tả và giá cả

3. **Tạo lớp Item** (items.py):
   - **Tokenization**: Sử dụng Meta-Llama-3.1-8B tokenizer
   - **Text cleaning**: Loại bỏ ký tự đặc biệt, thông tin không cần thiết
   - **Token constraints**: 
     - MIN_TOKENS: 150 (đảm bảo đủ thông tin)
     - MAX_TOKENS: 160 (giới hạn để tối ưu huấn luyện)
   - **Price filtering**: Chỉ giữ sản phẩm có giá từ $1-999

4. **Tối ưu hóa siêu tham số**:
   - **180 tokens**: Cân bằng giữa thông tin hữu ích và hiệu quả huấn luyện
   - Lý do chọn 180: Đủ context để đánh giá giá, nhưng không quá dài làm chậm huấn luyện

**Kết quả**:
- Dataset được làm sạch với ~X,000 items hợp lệ
- Mỗi item chứa: title, description, features, details, price
- Format chuẩn hóa cho các bước tiếp theo

#### Ngày 2: Mở Rộng Dataset và Feature Engineering

**Mục tiêu**: Tăng cường dữ liệu bằng cách thêm nhiều category sản phẩm và tạo features.

**Mở rộng dataset**:
- Thêm Electronics và Automotive
- Tăng tính đa dạng và khả năng tổng quát hóa
- Cân bằng phân phối giá giữa các category

**Feature Engineering**:
1. **Text-based features**:
   - Độ dài mô tả (text_length)
   - Số lượng từ khóa đặc biệt
   - Mật độ thông tin kỹ thuật

2. **Categorical features**:
   - Category encoding
   - Brand recognition (is_top_electronics_brand)
   - Rank trong category

3. **Numerical features**:
   - Weight/dimensions nếu có
   - Ratings và review counts

#### Ngày 3: Mô Hình Linear Regression Baseline

**Mục tiêu**: Xây dựng baseline model để so sánh với các phương pháp phức tạp hơn.

**Phương pháp**:
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Features được chọn
feature_columns = ['weight', 'rank', 'text_length', 'is_top_electronics_brand']

# Huấn luyện model
model = LinearRegression()
model.fit(X_train, y_train)

# Đánh giá
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

**Phương pháp đánh giá**:
- **Mean Squared Error (MSE)**: Đo sai số tuyệt đối
- **R² Score**: Phần trăm variance được giải thích
- **Root Mean Squared Log Error (RMSLE)**: Phù hợp với phân phối giá skewed

**Kết quả baseline**: Thiết lập chuẩn để so sánh các mô hình phức tạp hơn.

#### Ngày 4: Semantic Search và Retrieval-Augmented Generation (RAG)

**Mục tiêu**: Sử dụng semantic similarity để tìm sản phẩm tương tự và dự đoán giá.

**Phương pháp RAG**:
1. **Vector Database**: Tạo embeddings cho tất cả sản phẩm
2. **Similarity Search**: Tìm k sản phẩm tương tự nhất
3. **Price Aggregation**: Tính trung bình có trọng số từ sản phẩm tương tự

**Ưu điểm**:
- Không cần huấn luyện phức tạp
- Dễ hiểu và giải thích
- Hoạt động tốt với dữ liệu thưa

**Nhận xét kỹ thuật**: RAG approach cho phép leveraging existing data patterns mà không cần explicit training, phù hợp cho few-shot learning scenarios.

#### Ngày 5: Fine-tuning GPT-4o-mini

**Mục tiêu**: Sử dụng OpenAI API để fine-tune mô hình tiên tiến cho bài toán định giá.

**Chuẩn bị dữ liệu**:
```python
# Format JSONL cho OpenAI
fine_tune_train = train[:200]  # 200 examples
fine_tune_validation = train[200:250]  # 50 validation

# Prompt structure
prompt = f"How much does this cost?\n\n{product_description}\n\nPrice is $"
```

**Cấu hình Fine-tuning**:
```python
openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model="gpt-4o-mini-2024-07-18",
    seed=42,
    hyperparameters={"n_epochs": 1},
    suffix="pricer"
)
```

**Lý do chọn hyperparameters**:
- **1 epoch**: Tránh overfitting với dataset nhỏ
- **200 training examples**: Theo khuyến nghị OpenAI (50-100, nhưng tăng do example ngắn)
- **GPT-4o-mini**: Cân bằng giữa performance và cost

**Monitoring**: Sử dụng Weights & Biases để theo dõi training metrics.

### TUẦN 2: FINE-TUNING MÔ HÌNH MÃ NGUỒN MỞ

#### Ngày 1: Giới Thiệu QLoRA (Quantized Low-Rank Adaptation)

**Mục tiêu**: Hiểu về kỹ thuật QLoRA để fine-tune mô hình lớn hiệu quả.

**QLoRA Technical Deep-dive**:

1. **Quantization (4-bit)**:
   - Giảm memory footprint từ ~16GB xuống ~4GB
   - Sử dụng NF4 (Normal Float 4) quantization
   - Double quantization để tăng độ chính xác

2. **Low-Rank Adaptation (LoRA)**:
   - Thay vì update toàn bộ weights, chỉ train adapter matrices
   - Original weight W ≈ W + α * A * B
   - Với A (r × d), B (d × r), r << d (rank bottleneck)

3. **Target Modules**:
   ```python
   TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
   ```
   - Chỉ fine-tune attention projection layers
   - Giữ nguyên feed-forward networks

**Tính toán Memory**:
```python
# Base model: 8B parameters
# LoRA adapters: ~32M parameters (0.4% của base model)
# Memory reduction: 75% so với full fine-tuning
```

#### Ngày 2: Đánh Giá Mô Hình Gốc (Base Model Evaluation)

**Mục tiêu**: Thiết lập baseline performance của Llama 3.1 8B chưa fine-tune.

**Model Setup**:
```python
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
QUANT_4_BIT = True

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

**Evaluation Framework**:
```python
class Tester:
    def color_for(self, error, truth):
        if error < 40 or error/truth < 0.2:
            return "green"  # Good prediction
        elif error < 80 or error/truth < 0.4:
            return "orange"  # Acceptable
        else:
            return "red"    # Poor prediction
```

**Metrics**:
- **Average Error**: Sai số trung bình tuyệt đối ($)
- **RMSLE**: Root Mean Squared Log Error
- **Hit Rate**: % predictions trong ngưỡng chấp nhận được

#### Ngày 3-4: Huấn Luyện Với QLoRA

**Mục tiêu**: Fine-tune Llama 3.1 với QLoRA cho bài toán định giá.

**Hyperparameters**:
```python
# LoRA Configuration
LORA_R = 32              # Rank (độ phức tạp adapter)
LORA_ALPHA = 64          # Scaling factor
LORA_DROPOUT = 0.1       # Regularization
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training Configuration
EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
OPTIMIZER = "paged_adamw_32bit"
WARMUP_RATIO = 0.03
```

**Data Collator**:
```python
# Quan trọng: Chỉ tính loss cho phần response (giá)
# Mask phần input (product description) 
def create_prompt(item):
    question = "How much does this cost to the nearest dollar?"
    description = item['text']
    price = f"Price is ${item['price']:.0f}.00"
    return f"{question}\n\n{description}\n\n{price}"
```

**Training Process**:
1. **Supervised Fine-Tuning (SFT)**: Học predict token tiếp theo
2. **Gradient Accumulation**: Mô phỏng batch size lớn hơn
3. **Learning Rate Scheduling**: Cosine decay với warmup
4. **Monitoring**: Weights & Biases integration

**Kỹ thuật quan trọng**:
- **Instruction Tuning**: Mô hình học follow format "Question -> Description -> Price"
- **Causal Language Modeling**: Chỉ predict tokens phía sau, không bidirectional
- **Loss Masking**: Chỉ tính loss cho phần price prediction

#### Ngày 5: Đánh Giá Mô Hình Đã Fine-tune

**Mục tiêu**: So sánh hiệu suất trước và sau fine-tuning.

**Model Loading**:
```python
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config)
fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
```

**Advanced Prediction**:
```python
def improved_model_predict(prompt, top_K=3):
    # Lấy top-K tokens có xác suất cao nhất
    # Tính weighted average của predictions
    outputs = model(inputs)
    next_token_logits = outputs.logits[:, -1, :]
    top_probs, top_tokens = F.softmax(next_token_logits).topk(top_K)
    
    # Weighted combination
    weighted_price = sum(price * prob for price, prob in zip(prices, probs))
    return weighted_price
```

**So sánh kết quả**:
- **Base Model**: Baseline performance
- **Fine-tuned Model**: Improved accuracy
- **Improved Prediction**: Ensemble-like approach với top-K sampling

## Framework Đánh Giá (testing.py)

### Tester Class Architecture

```python
class Tester:
    def __init__(self, predictor, data, title=None, size=250):
        # Initialize evaluation framework
        
    def run_datapoint(self, i):
        # Evaluate single prediction
        error = abs(guess - truth)
        sle = (log(truth+1) - log(guess+1))^2
        
    def report(self):
        # Generate comprehensive evaluation report
        average_error = sum(errors) / size
        rmsle = sqrt(sum(sles) / size)
        hit_rate = hits / size * 100
```

### Metrics Explained

1. **Average Error ($)**: 
   - Ý nghĩa: Sai số trung bình tuyệt đối
   - Công thức: `Σ|predicted - actual| / n`
   - Thấp hơn = tốt hơn

2. **RMSLE (Root Mean Squared Log Error)**:
   - Ý nghĩa: Metric phù hợp với phân phối giá skewed  
   - Công thức: `√(Σ(log(predicted+1) - log(actual+1))² / n)`
   - Penalize relative errors hơn là absolute errors

3. **Hit Rate (%)**:
   - Green: Error < $40 hoặc < 20% giá trị thực
   - Orange: Error < $80 hoặc < 40% giá trị thực  
   - Red: Error ≥ $80 và ≥ 40% giá trị thực

## Kết Quả và Đánh Giá

### So Sánh Các Phương Pháp

| Method | Average Error | RMSLE | Hit Rate | Training Time | Memory Usage |
|--------|---------------|-------|----------|---------------|--------------|
| Random | ~$340 | ~1.72 | 11.6% | Minutes | Low |
| Human | ~$127 | 1 | 32% | Minutes | Low |
| Constant | $145.51 | 1.22 | 16.8% | Minutes | Low |
| Linear Reg + Features | ~$139.34 | 1.17 | 15.6% | Minutes | Low |
| Linear Reg + BagofWords | ~$113.6 | 0.99 | 24.8% | Minutes | Low |
| Linear Reg + word2vec | ~$116.76 | 1.07 | 24.4% | Minutes | Low |
| SVR + document_vector | ~$111.99 | 0.9 | 28% | Minutes | Medium |
| Random Forest | ~$97 | ~X.XX | ~X% | None | Medium |
| GPT-4o-mini | $79.58 | 0.59 | 52% | ~8 min | API |
| GPT-4o | $75.92 | 0.75 | 58% | ~10 min | API |
| Claude 3 | $100.83 | 0.6 | 50.8% | ~10 min | API |
| GPT Fine Tuned | $91.45 | 0.68 | 44% | ~20 min | API |
| Llama Base | $395.72 | 1.49 | 28% | None | 4GB |
| Llama Fine-tuned | $51.17 | 0.43 | 69.6% | ~6 hours | 4GB |

### Insights và Learning

1. **Data Quality is King**: 
   - Chất lượng dữ liệu quan trọng hơn mô hình phức tạp
   - Text preprocessing và feature engineering có impact lớn

2. **Model Size vs Performance**:
   - Mô hình lớn hơn không luôn tốt hơn cho specialized tasks
   - Fine-tuning focused beats general capabilities

3. **Efficiency Considerations**:
   - QLoRA enables fine-tuning với resource hạn chế
   - 4-bit quantization giảm 75% memory với minimal performance loss

4. **Prompt Engineering Matters**:
   - Format và structure của prompt ảnh hưởng lớn đến kết quả
   - Consistent formatting across train/test critical

## Setup và Installation

### Requirements

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc sử dụng conda
conda env create -f environment.yml
conda activate nlp-llm
```

### Environment Variables

Tạo file `.env`:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key  
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

### Hardware Requirements

- **Minimum**: 16GB RAM, GPU với 8GB VRAM
- **Recommended**: 32GB RAM, GPU với 16GB+ VRAM (A100, RTX 4090)
- **Cloud Options**: Google Colab Pro+, AWS/GCP với GPU instances

## Sử Dụng

### Chạy Evaluation

```python
from testing import Tester
from items import Item

# Load your model and data
# ...

# Run evaluation
Tester.test(your_prediction_function, test_data)
```

### Fine-tuning Mới

```python
# Xem week2/day3.ipynb để có full pipeline
# Hoặc chạy script:
python week2/week_2_day_3_training.py
```

## Tối Ưu Hóa và Cải Tiến

### Các Hướng Phát Triển

1. **Data Augmentation**:
   - Synthetic data generation
   - Cross-category transfer learning
   - Multi-modal features (images)

2. **Model Architecture**:
   - Ensemble methods
   - Multi-stage prediction pipeline
   - Uncertainty quantification

3. **Advanced Techniques**:
   - Parameter-efficient fine-tuning variants (AdaLoRA, QA-LoRA)
   - Knowledge distillation
   - Reinforcement Learning from Human Feedback (RLHF)

### Hyperparameter Tuning

```python
# Các tham số quan trọng cần tune:
LORA_R = [16, 32, 64]           # Model capacity
LEARNING_RATE = [1e-5, 1e-4, 1e-3]  # Convergence speed
BATCH_SIZE = [2, 4, 8]          # Stability vs speed
EPOCHS = [1, 2, 3]              # Overfitting balance
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Giảm batch_size
   - Sử dụng gradient_accumulation_steps
   - Enable gradient_checkpointing

2. **Poor Convergence**:
   - Kiểm tra learning rate
   - Thử warmup_ratio khác nhau
   - Verify data preprocessing

3. **Inconsistent Results**:
   - Set random seed
   - Check tokenizer consistency
   - Validate prompt format

## Đóng Góp và Phát Triển

### Code Structure

- `items.py`: Core data processing class
- `testing.py`: Evaluation framework  
- `loaders.py`: Data loading utilities
- `week*/`: Jupyter notebooks cho từng bước

### Best Practices

- Sử dụng type hints
- Document các magic numbers
- Version control cho experiments
- Reproducible seeds

## Tài Liệu Tham Khảo

### Papers
- QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

### Resources
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)

### Datasets
- [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

---

**Lưu ý**: Dự án này mang tính chất giáo dục và nghiên cứu. Kết quả có thể khác nhau tùy thuộc vào môi trường và cấu hình hardware.
