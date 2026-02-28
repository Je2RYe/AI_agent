# Completeness 评估方案对比

## 当前状况
- **短视频（3分钟）**: Completeness = 31-40% ✅
- **长视频（40分钟）**: Completeness = 12-14% ❌ 严重失真

## 三种方案对比

### 方案 Current（现有实现）
```python
# 固定 top_ratio=0.2
top_k = int(len(sentences) * 0.2)  # 长视频 → 400句！
# 单一向量匹配
summary_embed = encode([summary])  # 压缩成1个向量
similarities = cosine_similarity([summary_embed], key_embeds)
return (similarities > 0.4).mean()
```

**问题**：
- ❌ 长视频关键句爆炸（400句）
- ❌ 向量坍缩（多主题压缩成1个向量）
- ❌ 评分崩盘（12-14%）

---

### 方案 AI（外部建议）
```python
# 动态封顶
k_limit = min(int(n_total * 0.2), 50)  # 封顶50句
# 句子级N-vs-M
summary_embeds = encode(summary_sentences)  # N个句子向量
sim_matrix = cosine_similarity(summary_embeds, key_embeds)  # (N, M)
max_sim_per_key = sim_matrix.max(axis=0)
return (max_sim_per_key > 0.6).mean()
```

**优点**：
- ✅ 封顶50句，避免爆炸
- ✅ 句子级匹配，避免坍缩
- ✅ Max-Recall思想正确

**改进空间**：
- ⚠️ 阈值0.6偏严格（建议0.5）
- ⚠️ 封顶突变（100句→101句差异大）
- ⚠️ 只有单向Recall

---

### 方案 A++（推荐）
```python
def _get_dynamic_top_k(n_sentences):
    """平滑曲线，避免突变"""
    if n_sentences <= 100:
        return max(10, int(n_sentences * 0.20))
    elif n_sentences <= 500:
        return max(20, int(n_sentences * 0.15))
    else:
        return max(30, min(int(n_sentences * 0.05), 50))

def completeness(summary, transcript, threshold=0.5):
    # 1. 动态Top-K
    top_k = self._get_dynamic_top_k(len(transcript_sentences))
    
    # 2. 句子级编码
    summary_embeds = encode(summary_sentences)  # (N, dim)
    key_embeds = encode(key_sentences)          # (M, dim)
    
    # 3. N x M 矩阵，Max-Recall
    sim_matrix = cosine_similarity(summary_embeds, key_embeds)
    max_sim_per_key = sim_matrix.max(axis=0)
    
    # 4. 可选：双向验证（Recall为主，Precision为辅）
    recall = (max_sim_per_key > threshold).mean()
    precision = (sim_matrix.max(axis=1) > threshold).mean()
    score = 0.7 * recall + 0.3 * precision  # 可配置
    
    return score
```

**优点**：
- ✅ 平滑过渡（100句、500句分界点）
- ✅ 句子级Max-Recall
- ✅ 阈值0.5保守（可根据数据调优）
- ✅ 可选双向验证（面试加分项）
- ✅ 完整保留方案B拒绝理由

---

## 实施建议

### Phase 1: 核心修复（必须）
实现 **方案A++** 的动态Top-K + 句子级Recall

**预期效果**：
- 短视频：Completeness 维持在 30-40%
- 长视频：Completeness 提升到 25-35%（合理范围）

### Phase 2: 参数调优（可选）
根据实际数据调整：
```python
COMPLETENESS_CONFIG = {
    'threshold': 0.5,          # 相似度阈值
    'max_key_sentences': 50,   # 长视频封顶
    'recall_weight': 0.7,      # Recall vs Precision权重
}
```

### Phase 3: 双向验证（面试加分）
当 `precision < 0.4` 时触发告警（可能是幻觉）

---

## 面试话术（方案B为何不可行）

**问**: "为什么不用RAG的筛选context作为评估基准？"

**答**: 
"我保留了完整transcript作为ground truth，原因有三：

1. **评估独立性原则**：评估系统应独立于生成系统。用RAG输出评估RAG质量是循环论证。

2. **反向诊断能力**：当RAG场景Completeness下降时，我可以通过对比'摘要-RAG context-完整transcript'的三角关系，定位是**检索器召回不足**还是**生成器提炼能力弱**。这是工业界RAG评估的标准做法（参考Ragas框架的Context Precision/Recall分离思想）。

3. **多级监控方案**：
   - 用户侧：Completeness基于完整transcript（衡量信息损失）
   - 系统侧：RAG Context Coverage（监控检索质量）
   - 这样既保证评估严谨性，又能定位具体问题。

如果用方案B（RAG context作基准），本质是在'粉饰'评估结果，**掩盖了检索器的缺陷**。"

---

## 总结

**选择方案A++的理由**：
1. ✅ 解决长视频评估失真问题
2. ✅ 保持工业界标准（动态Top-K + Max-Recall）
3. ✅ 面试话术完整（技术+原理+应用）
4. ✅ 可扩展性强（参数可调优，支持双向验证）

**代码改动量**：
- 核心：~40行（动态Top-K函数 + completeness重写）
- 可选：+20行（双向验证）
- 配置：+5行（config.py添加参数）

**收益**：
- 长视频Completeness从12-14%提升到**25-35%**（合理范围）
- 评估系统可信度大幅提升
- 面试时技术深度+1
