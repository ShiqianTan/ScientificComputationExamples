由于完整的ChemReactSeek系统涉及多个复杂模块，我将提供一个简化的实现框架，包含核心功能的主要代码结构：

```python
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import json
from typing import List, Dict, Any
import openai

class ChemReactSeek:
    def __init__(self, knowledge_base_path: str, model_name: str = "deepseek-ai/deepseek-llm-7b-chat"):
        """
        初始化ChemReactSeek系统
        
        Args:
            knowledge_base_path: 知识库文件路径
            model_name: 使用的LLM模型名称
        """
        self.knowledge_base = self.load_knowledge_base(knowledge_base_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModel.from_pretrained(model_name)
        self.faiss_index = None
        self.text_corpus = []
        
        # 初始化向量数据库
        self._build_vector_database()
    
    def load_knowledge_base(self, file_path: str) -> List[Dict]:
        """加载化学知识库"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("知识库文件不存在，将创建空知识库")
            return []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件中提取文本"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def structured_data_extraction(self, text: str) -> Dict[str, Any]:
        """
        使用LLM从文本中提取结构化化学反应数据
        """
        prompt = f"""
        请从以下化学文献文本中提取反应信息，并按照指定格式输出：
        
        文本：
        {text}
        
        请提供提取的数据，格式如下：
        Title:
        Summary:
        Chemical Reaction:
        Experimental Conditions:
        Reagents:
        Solvent:
        Catalyst:
        Type of catalyst:
        Catalyst usage details:
        Pressure Range:
        Reaction Temperature:
        Reaction Time:
        pH Value:
        Specific Values:
        Yield Procedure:
        Expected Yield:
        Procedure: 1. 2. 3.
        Notes:
        """
        
        # 这里简化处理，实际应调用DeepSeek-v3 API
        extracted_data = self._call_llm_api(prompt)
        return self._parse_extracted_data(extracted_data)
    
    def _build_vector_database(self):
        """构建FAISS向量数据库"""
        if not self.knowledge_base:
            print("知识库为空，无法构建向量数据库")
            return
        
        # 准备文本语料
        self.text_corpus = []
        for item in self.knowledge_base:
            text_representation = f"""
            Title: {item.get('title', '')}
            Summary: {item.get('summary', '')}
            Reaction: {item.get('chemical_reaction', '')}
            Conditions: {item.get('experimental_conditions', '')}
            Catalyst: {item.get('catalyst', '')}
            Solvent: {item.get('solvent', '')}
            """
            self.text_corpus.append(text_representation)
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(self.text_corpus)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        
        # 归一化向量用于余弦相似度
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        print(f"向量数据库构建完成，包含 {len(self.text_corpus)} 个文档")
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        语义搜索相关反应信息
        
        Args:
            query: 查询文本
            k: 返回最相关的k个结果
        """
        if self.faiss_index is None:
            return []
        
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 搜索相似文档
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.knowledge_base):
                result = self.knowledge_base[idx].copy()
                result['similarity_score'] = float(similarities[0][i])
                results.append(result)
        
        return results
    
    def generate_reaction_protocol(self, query: str) -> Dict[str, Any]:
        """
        生成反应实验方案
        
        Args:
            query: 用户查询，如"设计HMF氢化反应的实验方案"
        """
        # 1. 检索相关知识
        retrieved_info = self.semantic_search(query)
        
        # 2. 构建增强提示
        context = self._build_rag_context(retrieved_info)
        
        # 3. 生成反应方案
        prompt = self._construct_protocol_prompt(query, context)
        protocol = self._call_llm_api(prompt)
        
        # 4. 解析和返回结果
        return {
            'query': query,
            'retrieved_context': retrieved_info,
            'generated_protocol': protocol,
            'evaluation_metrics': self._evaluate_protocol(protocol)
        }
    
    def _build_rag_context(self, retrieved_info: List[Dict]) -> str:
        """构建RAG上下文"""
        context = "相关文献信息：\n"
        for i, info in enumerate(retrieved_info, 1):
            context += f"\n--- 文献 {i} (相似度: {info.get('similarity_score', 0):.3f}) ---\n"
            context += f"标题: {info.get('title', '')}\n"
            context += f"摘要: {info.get('summary', '')}\n"
            context += f"催化剂: {info.get('catalyst', '')}\n"
            context += f"溶剂: {info.get('solvent', '')}\n"
            context += f"反应条件: {info.get('experimental_conditions', '')}\n"
        return context
    
    def _construct_protocol_prompt(self, query: str, context: str) -> str:
        """构建协议生成提示"""
        prompt = f"""
        你是一个专业的化学家助手。基于以下相关文献信息，为用户的查询设计详细、可行的实验方案。
        
        {context}
        
        用户查询：{query}
        
        请生成包含以下内容的详细实验方案：
        1. 推荐的催化剂及其用量
        2. 溶剂选择
        3. 反应条件（温度、压力、时间）
        4. 具体操作步骤
        5. 预期产率和选择性
        6. 安全注意事项
        7. 成本和经济性考虑
        
        请确保方案具体、可执行，并基于提供的文献信息。
        """
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        调用LLM API生成回复
        注意：这里需要根据实际使用的LLM服务进行实现
        """
        # 简化实现 - 实际应调用DeepSeek API或其他LLM服务
        try:
            # 示例：使用OpenAI格式的API调用
            response = openai.ChatCompletion.create(
                model="deepseek-chat",  # 或实际使用的模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM调用失败: {str(e)}"
    
    def _parse_extracted_data(self, text: str) -> Dict[str, Any]:
        """解析LLM提取的结构化数据"""
        # 实现文本到结构化数据的解析逻辑
        lines = text.split('\n')
        data = {}
        current_key = None
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()
                current_key = key.strip()
            elif current_key and line.strip():
                data[current_key] += ' ' + line.strip()
        
        return data
    
    def _evaluate_protocol(self, protocol: str) -> Dict[str, float]:
        """评估生成方案的质量"""
        # 简化的评估逻辑
        evaluation = {
            'feasibility_score': self._assess_feasibility(protocol),
            'safety_score': self._assess_safety(protocol),
            'cost_effectiveness_score': self._assess_cost_effectiveness(protocol),
            'green_chemistry_score': self._assess_green_chemistry(protocol)
        }
        return evaluation
    
    def _assess_feasibility(self, protocol: str) -> float:
        """评估可行性"""
        feasibility_keywords = ['明确', '具体', '可执行', '已验证', '文献支持']
        score = 0.0
        for keyword in feasibility_keywords:
            if keyword in protocol:
                score += 0.2
        return min(score, 1.0)
    
    def _assess_safety(self, protocol: str) -> float:
        """评估安全性"""
        safety_keywords = ['安全', '防护', '通风', '压力', '温度控制', '惰性气体']
        score = 0.0
        for keyword in safety_keywords:
            if keyword in protocol:
                score += 0.15
        return min(score, 1.0)
    
    def _assess_cost_effectiveness(self, protocol: str) -> float:
        """评估成本效益"""
        cost_keywords = ['廉价', '经济',可回收', '低成本', '工业级']
        score = 0.0
        for keyword in cost_keywords:
            if keyword in protocol:
                score += 0.2
        return min(score, 1.0)
    
    def _assess_green_chemistry(self, protocol: str) -> float:
        """评估绿色化学原则"""
        green_keywords = ['水相', '无毒', '可回收', '环境友好', '绿色溶剂']
        score = 0.0
        for keyword in green_keywords:
            if keyword in protocol:
                score += 0.2
        return min(score, 1.0)
    
    def add_literature(self, pdf_path: str):
        """添加新的文献到知识库"""
        text = self.extract_text_from_pdf(pdf_path)
        structured_data = self.structured_data_extraction(text)
        
        # 添加到知识库
        self.knowledge_base.append(structured_data)
        
        # 更新向量数据库
        self._build_vector_database()
        
        print(f"成功添加文献: {structured_data.get('title', '未知标题')}")

# 使用示例
def main():
    # 初始化系统
    chem_ai = ChemReactSeek("chemical_knowledge_base.json")
    
    # 查询示例
    query = "设计5-羟甲基糠醛(HMF)氢化生成2,5-二羟甲基呋喃的实验方案"
    
    # 生成反应方案
    result = chem_ai.generate_reaction_protocol(query)
    
    print("=== 生成的实验方案 ===")
    print(result['generated_protocol'])
    print("\n=== 评估指标 ===")
    for metric, score in result['evaluation_metrics'].items():
        print(f"{metric}: {score:.2f}")
    
    print("\n=== 检索到的相关文献 ===")
    for i, context in enumerate(result['retrieved_context'][:3], 1):
        print(f"{i}. {context.get('title', '')} (相似度: {context.get('similarity_score', 0):.3f})")

if __name__ == "__main__":
    main()
```

## 🔧 配套工具和配置文件

### 1. 知识库JSON结构示例 (`chemical_knowledge_base.json`)
```json
[
    {
        "title": "Palladium-catalyzed hydrogenation of furan derivatives",
        "summary": "Study on selective hydrogenation of furanic compounds using Pd/C catalyst",
        "chemical_reaction": "HMF → DHMF",
        "catalyst": "Pd/C (1-5 wt%)",
        "solvent": "methanol, ethanol, water",
        "experimental_conditions": "Temperature: 80-100°C, Pressure: 2-3 MPa, Time: 4-6 hours",
        "yield": "77%",
        "selectivity": "98%"
    }
]
```

### 2. 环境依赖 (`requirements.txt`)
```txt
faiss-cpu==1.7.4
torch==2.0.1
transformers==4.30.2
sentence-transformers==2.2.2
PyMuPDF==1.22.5
numpy==1.24.3
openai==0.27.8
```

## 💡 重要说明

1. **LLM API集成**：需要根据实际使用的LLM服务（如DeepSeek API）修改 `_call_llm_api` 方法
2. **向量数据库**：当前使用FAISS，可根据需要切换到其他向量数据库
3. **知识库构建**：需要预先构建化学文献知识库
4. **性能优化**：生产环境需要考虑缓存、批处理等优化措施

这个实现提供了ChemReactSeek的核心功能框架，实际部署时需要根据具体需求进行调整和优化。
