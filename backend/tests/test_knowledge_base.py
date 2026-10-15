"""知识库文档生命周期：同一文档再导入时替换旧版本、先写后删、种子增量同步、删除和列表。"""
import hashlib

import numpy as np
import pytest

import business.kb_seed as kb_seed
import retrieval.knowledge_base as kb_mod
from retrieval.knowledge_base import KnowledgeBase, document_id


class FakeEmbedder:
    """确定性的字符 n-gram 哈希向量，不加载模型；fail=True 时模拟向量 API 不可用。"""

    model_name = "fake-embedder"
    collection_suffix = "fake_embedder"

    def __init__(self):
        self.calls = 0
        self.fail = False

    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for i in range(len(text) - 1):
            v[int(hashlib.md5(text[i:i + 2].encode()).hexdigest(), 16) % 64] += 1.0
        return v / (np.linalg.norm(v) or 1.0)

    def __call__(self, input):  # noqa: A002 - ChromaDB 协议参数名
        if self.fail:
            raise ConnectionError("embedding api down")
        self.calls += len(input)
        return [self._vec(t) for t in input]

    def embed_query(self, text):
        return self._vec(text).tolist()


SEEDS = [
    {"title": "退款政策", "domain": "billing", "content": "服务未启动全额退。"},
    {"title": "发票说明", "domain": "billing", "content": "付清款项后 5 个工作日内开具。"},
]


@pytest.fixture
def seeds(monkeypatch):
    docs = [dict(d) for d in SEEDS]
    monkeypatch.setattr(kb_seed, "build_seed_documents", lambda: ("v1-" + str(len(docs)), [dict(d) for d in docs]))
    return docs


@pytest.fixture
def kb(tmp_path, monkeypatch, seeds):
    embedder = FakeEmbedder()
    monkeypatch.setattr(kb_mod, "get_shared_embedding_function", lambda: embedder)
    base = KnowledgeBase(chroma_host="", chroma_path=str(tmp_path))
    base.embedder = embedder
    return base


def _chunks(kb, title):
    return kb._collection.get(where={"title": title}, include=["documents", "metadatas"])


def test_reupload_replaces_old_version(kb):
    kb.add_documents([{"title": "定金规则补充", "content": "定金 2000 元，签约后 7 天内可全额退还。"}])
    kb.add_documents([{"title": "定金规则补充", "content": "定金 3000 元，签约后 3 天内可全额退还。"}])
    got = _chunks(kb, "定金规则补充")
    assert len(got["ids"]) == 1 and "3000" in got["documents"][0]
    assert [r["content"] for r in kb.search("定金多少", top_k=5) if r["title"] == "定金规则补充"] == [
        "定金 3000 元，签约后 3 天内可全额退还。"]


def test_unchanged_document_is_skipped(kb):
    doc = {"title": "说明", "content": "内容不变。"}
    assert kb.add_documents([doc]) == 1
    before = kb.embedder.calls
    assert kb.add_documents([doc]) == 0
    assert kb.embedder.calls == before


def test_fewer_chunks_removes_leftovers(kb):
    long = "。".join(f"第{i}句" + "很长的内容" * 20 for i in range(12))
    kb.add_documents([{"title": "长文", "content": long}])
    assert len(_chunks(kb, "长文")["ids"]) > 1
    kb.add_documents([{"title": "长文", "content": "现在只剩一句。"}])
    assert len(_chunks(kb, "长文")["ids"]) == 1


def test_failed_write_keeps_old_version(kb):
    kb.add_documents([{"title": "说明", "content": "旧内容。"}])
    kb.embedder.fail = True
    with pytest.raises(ConnectionError):
        kb.add_documents([{"title": "说明", "content": "新内容。"}])
    got = _chunks(kb, "说明")
    assert got["documents"] == ["【说明】旧内容。"]


def test_explicit_doc_id_allows_renaming(kb):
    kb.add_documents([{"title": "旧标题", "content": "A。", "doc_id": "policy-1"}])
    kb.add_documents([{"title": "新标题", "content": "A。", "doc_id": "policy-1"}])
    assert not _chunks(kb, "旧标题")["ids"] and _chunks(kb, "新标题")["ids"]


def test_seed_sync_is_incremental_and_removes_missing(kb, seeds):
    before = kb.embedder.calls
    assert kb.sync_seed_documents()["reseeded"] is False
    assert kb.embedder.calls == before

    kb.add_documents([{"title": "用户上传", "content": "不受种子同步影响。"}])
    seeds[0]["content"] = "服务未启动全额退；启动后按进度退。"
    del seeds[1]
    result = kb.sync_seed_documents()
    assert result["reseeded"] and result["chunks"] == 1 and result["removed"] == 1
    assert "按进度退" in _chunks(kb, "退款政策")["documents"][0]
    assert not _chunks(kb, "发票说明")["ids"]
    assert _chunks(kb, "用户上传")["ids"]


def test_seed_sync_failure_keeps_old_seeds(kb, seeds):
    seeds[0]["content"] = "改过的退款政策。"
    seeds.append({"title": "新文档", "domain": "shared", "content": "新内容。"})
    kb.embedder.fail = True
    with pytest.raises(ConnectionError):
        kb.sync_seed_documents()
    assert _chunks(kb, "退款政策")["documents"] == ["【退款政策】服务未启动全额退。"]
    assert _chunks(kb, "发票说明")["ids"]


def test_list_and_delete(kb):
    kb.add_documents([{"title": "上传文档", "content": "内容。", "domain": "consulting"}])
    docs = {d["title"]: d for d in kb.list_documents()}
    assert docs["上传文档"]["chunks"] == 1 and docs["上传文档"]["source"] == "upload"
    assert docs["退款政策"]["source"] == "seed"

    with pytest.raises(ValueError):
        kb.delete_document(docs["退款政策"]["doc_id"])
    assert kb.delete_document(docs["上传文档"]["doc_id"]) == 1
    assert kb.delete_document(docs["上传文档"]["doc_id"]) == 0


def test_old_chunks_get_doc_ids_and_are_replaced(tmp_path, monkeypatch, seeds):
    embedder = FakeEmbedder()
    monkeypatch.setattr(kb_mod, "get_shared_embedding_function", lambda: embedder)
    kb = KnowledgeBase(chroma_host="", chroma_path=str(tmp_path))
    # 模拟旧版本写入的片段：没有 doc_id / doc_version，ID 和正文绑定
    kb._collection.upsert(ids=["legacy-1"], documents=["【旧上传】旧内容。"],
                          metadatas=[{"title": "旧上传", "chunk_index": 0, "total_chunks": 1, "domain": "shared",
                                      "source": "upload"}])
    kb = KnowledgeBase(chroma_host="", chroma_path=str(tmp_path))
    meta = kb._collection.get(ids=["legacy-1"])["metadatas"][0]
    assert meta["doc_id"] == document_id("upload", "旧上传")

    kb.add_documents([{"title": "旧上传", "content": "新内容。"}])
    got = _chunks(kb, "旧上传")
    assert got["documents"] == ["【旧上传】新内容。"]
