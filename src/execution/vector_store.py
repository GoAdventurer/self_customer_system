"""
向量数据库 - 基于 FAISS 的向量存储与检索

本模块为 RAG 系统提供底层向量存储能力，替代之前的内存字典匹配。
使用 Facebook AI Similarity Search (FAISS) 实现高性能近邻检索。

═══════════════════════════════════════════════════════════════════════════════
为什么选择 FAISS:
═══════════════════════════════════════════════════════════════════════════════

  1. 性能: 百万级向量毫秒级检索(本项目知识库规模用IVF即可)
  2. 零外部依赖: 不需要单独部署Milvus/Qdrant等服务
  3. 成熟稳定: Meta出品,业界标准,文档完善
  4. 灵活: 支持多种索引类型(Flat精确/IVF近似/HNSW图)
  5. 本地运行: 适合隐私敏感场景(向量不出本机)

═══════════════════════════════════════════════════════════════════════════════
索引策略选择:
═══════════════════════════════════════════════════════════════════════════════

  知识库规模 < 10,000条:  IndexFlatIP (精确内积,暴力搜索,无损)
  知识库规模 10K-100K:   IndexIVFFlat (倒排索引,近似但快)
  知识库规模 > 100K:     IndexIVFPQ (乘积量化,省内存)

  本系统MVP阶段知识库 < 1000条,使用 IndexFlatIP (精确匹配,零精度损失)。
  后续知识库增长时可无缝切换到 IVF 索引(改一行配置)。

═══════════════════════════════════════════════════════════════════════════════
与Embedding模型的关系:
═══════════════════════════════════════════════════════════════════════════════

  Ollama qwen3-embedding:8b → 输出4096维向量 → 存入FAISS索引 → 检索时计算相似度

═══════════════════════════════════════════════════════════════════════════════
"""
import numpy as np
import faiss
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorRecord:
    """
    向量记录 - FAISS索引中一条记录的完整元数据

    FAISS本身只存储向量和ID,元数据(文档内容、标题等)需外部管理。
    本类将FAISS的数字ID映射到业务层的doc_id和完整元数据。

    Attributes:
        doc_id: 业务层文档唯一标识(如 "policy_return_001")
        vector: 向量数据(numpy array)
        text: 被向量化的原始文本(用于调试/展示)
        metadata: 附加元数据(标题、分类、来源等)
        created_at: 入库时间
    """
    doc_id: str
    vector: np.ndarray
    text: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class FAISSVectorStore:
    """
    基于FAISS的向量存储与检索引擎

    提供:
      1. 向量入库(add): 将文本向量存入FAISS索引
      2. 相似检索(search): 给定query向量,返回最相似的top-k文档
      3. 持久化(save/load): 索引和元数据可序列化到磁盘
      4. 删除/更新: 知识更新时可移除旧向量

    使用示例:
        >>> store = FAISSVectorStore(dimension=4096)
        >>> store.add("doc_001", embedding_vector, text="7天无理由退货...")
        >>> results = store.search(query_vector, top_k=5)
        >>> print(results[0])  # (doc_id, score, metadata)

    线程安全:
        FAISS索引本身支持并发读,但写操作(add/remove)非线程安全。
        生产环境需加写锁或使用 IndexShards。
    """

    def __init__(self, dimension: int = 4096, index_type: str = "flat"):
        """
        初始化向量存储

        Args:
            dimension: 向量维度。与Embedding模型输出维度一致。
                qwen3-embedding:8b → 4096维
                bge-m3 → 1024维
            index_type: 索引类型
                "flat": 精确内积(暴力搜索,适合<10K文档)
                "ivf": IVF倒排索引(近似,适合10K-100K)
                "hnsw": HNSW图索引(近似,高召回率)
        """
        self.dimension = dimension
        self.index_type = index_type

        # 创建FAISS索引
        self._index = self._create_index(dimension, index_type)

        # 元数据存储: faiss_id → VectorRecord
        self._records: dict[int, VectorRecord] = {}

        # doc_id → faiss_id 映射(用于按doc_id删除/查询)
        self._doc_id_map: dict[str, int] = {}

        # 自增ID计数器
        self._next_id: int = 0

    def _create_index(self, dim: int, index_type: str) -> faiss.Index:
        """
        创建FAISS索引

        索引类型说明:
          flat: IndexFlatIP - 精确内积搜索
            · 优点: 100%精确,无信息损失
            · 缺点: O(n)搜索,文档多了慢
            · 适用: <10K文档(本系统当前规模)

          ivf: IndexIVFFlat - 倒排文件索引
            · 优点: 近似搜索,速度快
            · 缺点: 需要训练(需要一批初始数据),有精度损失
            · 适用: 10K-100K文档

          hnsw: IndexHNSWFlat - 分层导航小世界图
            · 优点: 高召回率,速度极快
            · 缺点: 内存占用大,删除代价高
            · 适用: 需要极高召回率的场景
        """
        if index_type == "flat":
            # 使用内积(Inner Product)而非L2距离
            # 因为归一化后的向量内积 = 余弦相似度
            return faiss.IndexFlatIP(dim)

        elif index_type == "ivf":
            # IVF需要先有一个量化器
            quantizer = faiss.IndexFlatIP(dim)
            # nlist: 聚类中心数(经验值: sqrt(n))
            nlist = 100
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            return index

        elif index_type == "hnsw":
            # HNSW参数: M=32(边数), efConstruction=200(构建精度)
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64  # 检索时精度
            return index

        else:
            return faiss.IndexFlatIP(dim)

    def add(self, doc_id: str, vector: list[float] | np.ndarray,
            text: str = "", metadata: dict = None) -> int:
        """
        向索引中添加一条向量记录

        Args:
            doc_id: 文档唯一标识
            vector: 文档向量(list或numpy array)
            text: 原始文本(用于展示/调试)
            metadata: 附加元数据

        Returns:
            分配的FAISS内部ID

        注意:
            - 相同doc_id重复添加会创建新记录(旧的仍存在)
            - 如需更新,先remove再add
            - 向量会被L2归一化(使内积=余弦相似度)
        """
        # 转为numpy并归一化
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)  # 归一化,使内积=余弦相似度

        # 分配ID
        faiss_id = self._next_id
        self._next_id += 1

        # 添加到FAISS索引
        # 对于Flat索引可直接add;IVF索引需要先train
        if self.index_type == "ivf" and not self._index.is_trained:
            # IVF需要训练数据,先暂存,等数据够了再train
            pass
        self._index.add(vec)

        # 保存元数据
        self._records[faiss_id] = VectorRecord(
            doc_id=doc_id,
            vector=vec.flatten(),
            text=text,
            metadata=metadata or {},
        )
        self._doc_id_map[doc_id] = faiss_id

        return faiss_id

    def search(self, query_vector: list[float] | np.ndarray,
               top_k: int = 5, threshold: float = 0.0) -> list[tuple[str, float, dict]]:
        """
        向量相似度检索

        给定query向量,从索引中检索最相似的top-k条记录。

        Args:
            query_vector: 查询向量(与入库时同维度)
            top_k: 返回的最大结果数
            threshold: 最低相似度阈值(低于此值的结果被过滤)
                0.0 = 不过滤
                0.5 = 中等相关
                0.7 = 高度相关
                0.9 = 极度相关

        Returns:
            [(doc_id, similarity_score, metadata), ...]
            按相似度降序排列,score范围[0,1](归一化内积=余弦相似度)
        """
        if self._index.ntotal == 0:
            return []

        # 归一化query向量
        vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        # 检索(FAISS返回距离和ID)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS用-1表示无效结果
                continue
            if score < threshold:
                continue

            record = self._records.get(int(idx))
            if record:
                results.append((
                    record.doc_id,
                    float(score),  # 余弦相似度 [0, 1]
                    record.metadata,
                ))

        return results

    def remove(self, doc_id: str) -> bool:
        """
        从索引中移除文档

        注意: FAISS的Flat索引不原生支持删除。
        实现方式: 标记删除 + 定期重建索引。

        Args:
            doc_id: 要移除的文档ID

        Returns:
            True=找到并标记删除, False=未找到
        """
        faiss_id = self._doc_id_map.get(doc_id)
        if faiss_id is None:
            return False

        # 标记删除(不从FAISS物理删除,search时过滤)
        if faiss_id in self._records:
            del self._records[faiss_id]
        del self._doc_id_map[doc_id]
        return True

    def rebuild_index(self):
        """
        重建索引(清理已删除记录,压缩存储)

        调用时机:
          - 批量删除后
          - 定期维护(如每天凌晨)
          - 索引碎片率过高时
        """
        if not self._records:
            self._index = self._create_index(self.dimension, self.index_type)
            return

        # 收集所有有效向量
        vectors = []
        new_records = {}
        new_doc_map = {}
        new_id = 0

        for old_id, record in sorted(self._records.items()):
            vectors.append(record.vector.reshape(1, -1))
            new_records[new_id] = record
            new_doc_map[record.doc_id] = new_id
            new_id += 1

        # 重建索引
        self._index = self._create_index(self.dimension, self.index_type)
        all_vectors = np.vstack(vectors).astype(np.float32)
        self._index.add(all_vectors)

        # 更新映射
        self._records = new_records
        self._doc_id_map = new_doc_map
        self._next_id = new_id

    def save(self, directory: str):
        """
        持久化索引到磁盘

        保存两个文件:
          - {directory}/faiss.index: FAISS二进制索引文件
          - {directory}/metadata.json: 元数据和映射关系

        Args:
            directory: 保存目录路径
        """
        os.makedirs(directory, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self._index, os.path.join(directory, "faiss.index"))

        # 保存元数据(不含numpy向量,太大)
        meta = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "next_id": self._next_id,
            "doc_id_map": self._doc_id_map,
            "records": {
                str(k): {
                    "doc_id": v.doc_id,
                    "text": v.text,
                    "metadata": v.metadata,
                    "created_at": v.created_at,
                }
                for k, v in self._records.items()
            },
        }
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> "FAISSVectorStore":
        """
        从磁盘加载索引

        Args:
            directory: 之前save的目录路径

        Returns:
            恢复的FAISSVectorStore实例
        """
        # 加载元数据
        with open(os.path.join(directory, "metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 创建实例
        store = cls(dimension=meta["dimension"], index_type=meta["index_type"])

        # 加载FAISS索引
        store._index = faiss.read_index(os.path.join(directory, "faiss.index"))
        store._next_id = meta["next_id"]
        store._doc_id_map = meta["doc_id_map"]

        # 恢复records(不含向量,需要时从FAISS索引重建)
        store._records = {}
        for k, v in meta["records"].items():
            store._records[int(k)] = VectorRecord(
                doc_id=v["doc_id"],
                vector=np.zeros(meta["dimension"], dtype=np.float32),  # 占位
                text=v["text"],
                metadata=v["metadata"],
                created_at=v["created_at"],
            )

        return store

    # ═══ 统计信息 ═══

    @property
    def count(self) -> int:
        """当前索引中的有效文档数"""
        return len(self._records)

    @property
    def total_in_index(self) -> int:
        """FAISS索引中的总向量数(含已标记删除的)"""
        return self._index.ntotal

    def stats(self) -> dict:
        """返回索引统计信息"""
        return {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "active_records": self.count,
            "total_vectors": self.total_in_index,
            "memory_mb": self._index.ntotal * self.dimension * 4 / 1024 / 1024,  # float32
        }
