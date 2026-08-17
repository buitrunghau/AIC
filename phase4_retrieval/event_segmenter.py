import torch
import warnings
from transformers import AutoTokenizer
from .model import EventSegmentationBERT_CRF

# Tắt cảnh báo
warnings.filterwarnings("ignore")


class EventSegmenter:
    def __init__(self, model_dir: str = "phobert_event_segmentation"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # 2. Khởi tạo kiến trúc Model (Phải khớp 100% với file train.py)
        # Giả định MAX_EVENTS = 10, nên num_labels = 1 (cho 'O') + 10*2 (B/I) = 21
        self.num_labels = 21

        self.model = EventSegmentationBERT_CRF(
            model_name="vinai/phobert-base-v2",
            num_labels=self.num_labels,
            lstm_hidden_size=256,
            lstm_layers=1,
            n_bert_layers=4,
        ).to(self.device)

        # 3. Load trọng số đã huấn luyện
        model_path = f"{model_dir}/pytorch_model.bin"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Ánh xạ từ Label ID sang Tên (B-Event1, I-Event1...)
        self.id2tag = {0: "O"}
        for i in range(1, 11):
            self.id2tag[len(self.id2tag)] = f"B-Event{i}"
            self.id2tag[len(self.id2tag)] = f"I-Event{i}"

    def segment(self, query: str) -> list:
        """
        Nhận vào câu truy vấn, trả về mảng các sự kiện con được tách ra.
        Ví dụ: "Tôi ăn cơm rồi đi ngủ" -> ["Tôi ăn cơm", "đi ngủ"]
        """
        if not query.strip():
            return []

        # Tokenize (Cần lấy offset_mapping để cắt chuỗi gốc cho chuẩn)
        encoded = self.tokenizer(
            query,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=128
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        offsets = encoded["offset_mapping"][0].tolist()  # [(0,0), (0,3), (4,6)...]

        # Inference qua mô hình (Viterbi Decoding từ CRF)
        with torch.no_grad():
            prediction = self.model(input_ids, attention_mask)

        # prediction[0] chứa mảng các label IDs đã được CRF tính toán đường đi tốt nhất
        pred_tags = [self.id2tag[p] for p in prediction[0]]

        # Giải mã các thẻ B-I thành chuỗi (String) gốc
        events = []
        current_event_chars = []

        for idx, (tag, offset) in enumerate(zip(pred_tags, offsets)):
            # Bỏ qua các token đặc biệt <s>, </s>, <pad> (có offset là 0,0)
            if offset[0] == 0 and offset[1] == 0:
                continue

            start_char, end_char = offset

            if tag.startswith("B-"):
                # Bắt đầu một sự kiện mới
                if current_event_chars:
                    # Lưu sự kiện trước đó (nếu có)
                    events.append(self._extract_text(query, current_event_chars))
                current_event_chars = [offset]

            elif tag.startswith("I-"):
                # Nối tiếp sự kiện hiện tại
                if current_event_chars:
                    current_event_chars.append(offset)

            elif tag == "O":
                # Kết thúc chuỗi sự kiện hiện tại
                if current_event_chars:
                    events.append(self._extract_text(query, current_event_chars))
                    current_event_chars = []

        # Bắt sự kiện cuối cùng (nếu câu kết thúc bằng I-Event)
        if current_event_chars:
            events.append(self._extract_text(query, current_event_chars))

        return events

    def _extract_text(self, original_text: str, offset_list: list) -> str:
        """Ghép các offset liên tiếp lại để cắt nguyên vẹn chuỗi gốc"""
        min_start = min(os[0] for os in offset_list)
        max_end = max(os[1] for os in offset_list)
        return original_text[min_start:max_end].strip()
