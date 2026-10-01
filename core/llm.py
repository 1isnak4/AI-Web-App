import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class RAGChatbot:
    def __init__(self, embed_model_name, llm_model_name, device="cpu"):
        self.device = device
        self.embedder = SentenceTransformer(embed_model_name)
        self.chunks = []
        self.index = None
        
        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )

    def build_knowledge_base(self, docs: list):
        """docs: danh sách các đoạn văn bản chính sách (chunks)"""
        self.chunks = docs
        embeddings = self.embedder.encode(docs, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype('float32'))

    def retrieve(self, query: str, top_k=3):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        scores, indices = self.index.search(q_emb.astype('float32'), top_k)
        
        retrieved_docs = [self.chunks[i] for i in indices[0] if i != -1]
        return retrieved_docs

    def generate_response(self, query: str):
        context_docs = self.retrieve(query)
        context_str = "\n---\n".join(context_docs)
        
        prompt = f"""Bạn là trợ lý tư vấn khách hàng của cửa hàng ShopLite. Hãy trả lời câu hỏi dựa trên thông tin chính sách dưới đây. Nếu không tìm thấy thông tin, hãy lịch sự từ chối.

Thông tin tham khảo:
{context_str}

Câu hỏi: {query}
Trả lời:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {
            "answer": response.strip(),
            "sources": context_docs
        }
