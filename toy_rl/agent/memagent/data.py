"""A1: MemAgent 数据源 —— 镜像源 agentic/memagent/prepare_data.py 的产物（nano 硬编码 mini QA）。

源用 prepare_data.py 从 HotpotQA/RULER 生成 JSONL：`{prompt, label, metadata:{context}}`（源 rollout.py:14-20）。
nano 只需几条能跑通闭环的**短文档**：context 足够长可切成 2-3 个 chunk，答案埋在其中一个 chunk 里，
逼记忆循环真的"逐 chunk 读、更新记忆、最后据记忆回答"。

偏离（详见 docs/decisions/a1.md）：硬编码 mini QA 而非真数据集——结构等价（prompt/label/context 三件套），
真 HotpotQA/RULER 数据留后续。为让答案能被 is_equiv（去空格/小写）稳定命中，label 取单 token 短答案。
"""

from __future__ import annotations

from toy_rl.sample import Sample

# 每条：问题 + 答案 + 一段可切成多 chunk 的短文档。答案句子刻意放在文档中后段，
# 前面塞干扰句，逼"逐 chunk 更新记忆"而非只看第一个 chunk。
_QA: list[dict] = [
    {
        "prompt": "In which city is the Everbright Tower located?",
        "label": "Lyonar",
        "context": (
            "The Coastal Survey of 1904 catalogued dozens of lighthouses along the northern reefs. "
            "Trade in salted fish dominated the regional economy for over two centuries. "
            "Many early maps of the delta were drawn by cartographers from distant guilds. "
            "The river port handled grain, timber, and later machine parts. "
            "The Everbright Tower, completed in 1928, stands in the city of Lyonar on the western hill. "
            "It was designed as a civic clock tower and remains a landmark today. "
            "Later renovations added an observation deck open to visitors each summer."
        ),
    },
    {
        "prompt": "What material is the Kestrel Bridge mainly built from?",
        "label": "granite",
        "context": (
            "The valley was first settled by shepherds who followed the seasonal grasses. "
            "A network of footpaths connected the highland villages before any roads existed. "
            "Floods in the spring often cut off the lower hamlets for weeks. "
            "Engineers proposed several crossings, but funding delayed construction for a decade. "
            "The Kestrel Bridge, opened in 1873, is built mainly from local granite quarried nearby. "
            "Its three arches have survived numerous floods without structural damage. "
            "Today it carries a footpath and a narrow service lane."
        ),
    },
    {
        "prompt": "Who composed the Aurora Symphony?",
        "label": "Velasquez",
        "context": (
            "The conservatory archive holds thousands of manuscripts from the romantic period. "
            "Several works were long attributed to anonymous court musicians. "
            "Concert programs from that era rarely credited living composers by full name. "
            "A ledger discovered in 1990 clarified many disputed authorships. "
            "The Aurora Symphony was composed by Velasquez in the final year of a long career. "
            "It premiered to a small audience but later became a staple of the repertoire. "
            "Critics praised its restrained use of brass and its long, unbroken melodic lines."
        ),
    },
    {
        "prompt": "What year was the Meridian Observatory founded?",
        "label": "1867",
        "context": (
            "Astronomy in the region began with amateur stargazers on the coastal cliffs. "
            "Cloud cover made systematic observation difficult for much of the year. "
            "A wealthy patron funded the first permanent instruments after a famous comet passed. "
            "Debate over the site delayed the project through several administrations. "
            "The Meridian Observatory was founded in 1867 on the granite plateau above the town. "
            "Its original telescope is now displayed in the entrance hall. "
            "Modern light pollution has since pushed active research to a darker mountain site."
        ),
    },
]


def load_data_source(args) -> list[Sample]:
    """对齐源 data_source_cls(args)：返回带 metadata['context'] 的 Sample 列表（MemAgent 路径）。"""
    return [
        Sample(prompt=q["prompt"], label=q["label"], metadata={"context": q["context"]})
        for q in _QA
    ]
