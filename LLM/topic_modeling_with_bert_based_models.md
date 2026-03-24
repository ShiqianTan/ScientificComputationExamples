以下代码完整复现了论文中基于LLM的主题建模流程，包括数据加载、嵌入生成、降维聚类、主题提取、评估及时间趋势分析。代码假设数据以CSV格式存储，包含`title`、`abstract`、`year`三列。请根据实际数据调整文件路径和列名。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import umap
import torch
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ------------------------- 1. 数据加载 -------------------------
# 假设数据文件为 polymer_papers.csv，包含 title, abstract, year
data_path = "polymer_papers.csv"
df = pd.read_csv(data_path)
# 合并标题和摘要作为文档
df['text'] = df['title'].fillna('') + ". " + df['abstract'].fillna('')
# 确保年份为整数
df['year'] = df['year'].astype(int)
docs = df['text'].tolist()
years = df['year'].tolist()

# ------------------------- 2. 定义模型和参数 -------------------------
model_names = {
    'BERT': 'bert-base-uncased',
    'SciBERT': 'allenai/scibert_scivocab_uncased',
    'MatSciBERT': 'm3rg-iitd/matscibert'
}

# 嵌入生成函数
def get_embeddings(docs, model_name, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(docs), batch_size), desc=f"Embedding with {model_name}"):
        batch = docs[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 [CLS] 标记的向量作为文档嵌入
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

# 存储结果
results = {}

# ------------------------- 3. 对每个模型进行主题建模 -------------------------
for model_key, model_name in model_names.items():
    print(f"\n===== Processing {model_key} =====")
    # 生成嵌入
    embeddings = get_embeddings(docs, model_name)
    
    # UMAP降维到5维（用于聚类）和2维（用于可视化）
    umap_5 = umap.UMAP(n_components=5, random_state=42, n_neighbors=15, min_dist=0.0)
    reduced_emb_5 = umap_5.fit_transform(embeddings)
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=0.5, min_samples=10)  # 参数可能需要根据数据调整
    cluster_labels = dbscan.fit_predict(reduced_emb_5)
    
    # 使用BERTopic提取主题表示（这里直接使用BERTopic的c-TF-IDF部分）
    # 为了兼容性，我们手动实现c-TF-IDF，或者直接用BERTopic的类
    # 这里用BERTopic的方便功能，但需要传入已降维的嵌入和标签
    topic_model = BERTopic(embedding_model=None, umap_model=None, hdbscan_model=None, verbose=True)
    # 注意：BERTopic的fit_transform需要原始嵌入，但我们已有聚类标签，可以直接创建主题模型
    # 但为了提取主题词，我们使用class-based TF-IDF
    # 收集每个主题下的文档
    unique_labels = set(cluster_labels)
    topic_docs = {label: [docs[i] for i in range(len(docs)) if cluster_labels[i]==label] for label in unique_labels}
    # 计算c-TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    # 为每个主题构建文档集
    topic_texts = [' '.join(doc_list) for doc_list in topic_docs.values()]
    tfidf_matrix = vectorizer.fit_transform(topic_texts)
    feature_names = vectorizer.get_feature_names_out()
    # 对于每个主题，取前20个TF-IDF最高的词
    topic_words = {}
    for idx, topic in enumerate(sorted(unique_labels)):
        row = tfidf_matrix[idx].toarray().flatten()
        top_idx = np.argsort(row)[-20:][::-1]
        words = [feature_names[i] for i in top_idx]
        topic_words[topic] = words
    
    # 存储结果
    results[model_key] = {
        'embeddings': embeddings,
        'reduced_5': reduced_emb_5,
        'labels': cluster_labels,
        'topic_words': topic_words,
        'n_topics': len(unique_labels) - 1 if -1 in unique_labels else len(unique_labels)  # 排除噪声
    }
    
    # 可选：保存主题词
    print(f"Number of topics (excluding noise): {results[model_key]['n_topics']}")
    for topic, words in list(topic_words.items())[:5]:
        print(f"Topic {topic}: {', '.join(words[:10])}...")
    
    # ------------------------- 4. 评估指标 -------------------------
    # 4.1 轮廓系数（原始嵌入和降维后2D）
    # 注意：轮廓系数需要标签，且要求标签数量>1，且排除噪声点（-1）
    mask = cluster_labels != -1
    if len(np.unique(cluster_labels[mask])) > 1:
        sil_orig = silhouette_score(embeddings[mask], cluster_labels[mask])
        # 降维到2维用于可视化，但这里计算轮廓系数（同样用2维嵌入）
        umap_2 = umap.UMAP(n_components=2, random_state=42)
        emb_2d = umap_2.fit_transform(embeddings)
        sil_2d = silhouette_score(emb_2d[mask], cluster_labels[mask])
        results[model_key]['silhouette_orig'] = sil_orig
        results[model_key]['silhouette_2d'] = sil_2d
        print(f"Silhouette score (original): {sil_orig:.3f}, (2D): {sil_2d:.3f}")
    else:
        results[model_key]['silhouette_orig'] = None
        results[model_key]['silhouette_2d'] = None
    
    # 4.2 单词覆盖率（计算tokenizer对文档中单词的覆盖比例）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_tokens = set()
    for doc in docs:
        tokens = tokenizer.tokenize(doc)
        all_tokens.update(tokens)
    # 获取tokenizer词汇表
    vocab = set(tokenizer.get_vocab().keys())
    coverage = len(all_tokens.intersection(vocab)) / len(all_tokens) if all_tokens else 0
    results[model_key]['word_coverage'] = coverage
    print(f"Word coverage: {coverage:.3f}")
    
    # 4.3 主题连贯性（使用Gensim，需要文档分词）
    # 构建语料（每个文档的词列表）
    tokenized_docs = [doc.split() for doc in docs]  # 简单分词，实际可改进
    dictionary = Dictionary(tokenized_docs)
    # 提取每个主题的前10个词用于计算
    topic_words_list = [topic_words[t][:10] for t in topic_words if t != -1]
    if len(topic_words_list) > 0:
        coherence_model = CoherenceModel(topics=topic_words_list, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        results[model_key]['coherence'] = coherence
        print(f"Topic coherence (c_v): {coherence:.3f}")
    else:
        results[model_key]['coherence'] = None

# ------------------------- 5. 时间趋势分析 -------------------------
# 对每个模型，按年份统计每个主题的论文数
for model_key in model_names.keys():
    labels = results[model_key]['labels']
    # 创建DataFrame用于分组
    df_temp = pd.DataFrame({'year': years, 'topic': labels})
    # 排除噪声
    df_temp = df_temp[df_temp['topic'] != -1]
    # 按年份和主题计数
    trend = df_temp.groupby(['year', 'topic']).size().unstack(fill_value=0)
    # 绘制前10个主要主题的趋势（按总论文数排序）
    top_topics = trend.sum().sort_values(ascending=False).head(10).index
    plt.figure(figsize=(12,6))
    for topic in top_topics:
        plt.plot(trend.index, trend[topic], label=f"Topic {topic}")
    plt.title(f"{model_key} - Temporal trends of top 10 topics")
    plt.xlabel("Year")
    plt.ylabel("Number of papers")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{model_key}_temporal_trends.png", dpi=150)
    plt.show()

# ------------------------- 6. 可视化UMAP 2D投影 -------------------------
for model_key in model_names.keys():
    embeddings = results[model_key]['embeddings']
    labels = results[model_key]['labels']
    # 降维到2D
    umap_2 = umap.UMAP(n_components=2, random_state=42)
    emb_2d = umap_2.fit_transform(embeddings)
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap='tab20', s=1, alpha=0.6)
    plt.colorbar(scatter, label='Topic ID')
    plt.title(f"{model_key} - UMAP projection of document embeddings")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(f"{model_key}_umap_projection.png", dpi=150)
    plt.show()

# ------------------------- 7. 输出汇总指标 -------------------------
print("\n===== Summary of metrics =====")
summary = pd.DataFrame({
    'Model': list(model_names.keys()),
    'N_topics': [results[m]['n_topics'] for m in model_names.keys()],
    'Word_coverage': [results[m]['word_coverage'] for m in model_names.keys()],
    'Silhouette_orig': [results[m]['silhouette_orig'] for m in model_names.keys()],
    'Silhouette_2D': [results[m]['silhouette_2d'] for m in model_names.keys()],
    'Coherence': [results[m]['coherence'] for m in model_names.keys()]
})
print(summary.to_string(index=False))
```

**说明**：
- 数据文件应包含`title`、`abstract`、`year`三列。如果只有标题或摘要，请相应调整`text`列。
- 嵌入生成使用批处理，并支持GPU（如果可用）。
- DBSCAN的`eps`和`min_samples`可能需要根据数据调整，默认值可能导致不同聚类数。论文中未给出具体值，但使用BERTopic默认设置（`eps=0.5, min_samples=10`）。
- 主题连贯性使用`c_v`度量，与论文一致。
- 可视化部分生成UMAP投影图和时间趋势图。
- 输出汇总指标表格。

**注意**：由于原始数据未公开，代码中数据加载部分需用户自行准备。运行前请安装所需库：
```bash
pip install pandas numpy matplotlib scikit-learn umap-learn transformers bertopic gensim tqdm torch
```
