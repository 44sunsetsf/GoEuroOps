"""多轮记忆评测：记忆模块能不能让模型记住、记对用户说过的话。

考点参照 LongMemEval 的分类，按本项目的场景改写，三套题：
  basic（20 段）
    early_recall   第 1 轮说的事实，5–7 轮后再问
    post_compress  第 1 轮说的事实，9 轮以上后再问（已经滑出最近对话，只能靠笔记）
    update         中途改了主意，问现在的值（旧值不能当成答案）
    history        中途改了主意，问最开始的值
    cross_session  在一个会话里说，在新会话里问
    abstention     从没说过的事，应该回答“未知”，不能编
  hard（10 段）：多个事实、跨多个会话改主意、长消息、长对话
  stress（10 段，单独跑）：闲聊陈述句也会存成笔记（含“表姐在荷兰”这类干扰项），多会话、反复修改、相对时间、多跳

怎么跑：
  - 用项目自己的 MemoryManager，每轮按对话接口的顺序写入用户消息、助手回复，再调用 after_turn；
    会话之间调用 enrich_notes，相当于线上用户停下来 IDLE_SECONDS 后的补全；
  - Redis 换成进程内的假实现（只实现用到的几个命令），向量库用临时目录的嵌入式 ChromaDB，向量模型和线上一样；
  - 笔记补全和最后的提问都调用真实模型（读 ANTHROPIC_* 环境变量）；
  - 提问时用 get_context() 拼出的记忆文本，让模型只依据它回答一个短语，按关键词判分。

用法：python -m evaluation.memory_benchmark --runs 3 --label amem
      python -m evaluation.memory_benchmark --runs 3 --set stress --label stress-amem
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_utils import NO_THINKING_KWARGS, extract_text_content, make_client  # noqa: E402
from memory.amem import MemoryManager, MsgRole  # noqa: E402

REPORT_DIR = ROOT / "evaluation" / "reports"


class FakeRedis:
    """进程内的 Redis 替身，只实现记忆模块用到的命令；lpush 最新在前，和真实 Redis 一致。"""

    def __init__(self) -> None:
        self.lists: Dict[str, List[str]] = {}
        self.kv: Dict[str, str] = {}

    async def lpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    async def expire(self, *args: Any) -> bool:
        return True

    async def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        self.lists[key] = self.lists.get(key, [])[start:end + 1]
        return True

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        return self.lists.get(key, [])[start:end + 1]

    async def delete(self, key: str) -> int:
        return int(self.lists.pop(key, None) is not None or self.kv.pop(key, None) is not None)

    async def get(self, key: str) -> Optional[str]:
        return self.kv.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self.kv[key] = value
        return True

    async def set(self, key: str, value: str, *args: Any, **kwargs: Any) -> bool:
        self.kv[key] = value
        return True

    async def aclose(self) -> None:
        return None


# ── 对话素材 ──────────────────────────────────────────────────────────────────

# 填充轮：真实咨询里常见的问答，助手回复长度贴近线上（几百字），用来把早期事实“挤”远
FILLERS: List[Tuple[str, str]] = [
    ("瑞典的硕士一般读几年？",
     "瑞典的硕士项目一般是两年制，按 120 个 ECTS 学分计算。第一年以专业必修课为主，第二年上半学期是选修课和项目，下半学期写毕业论文，"
     "论文通常可以在企业里完成，比如爱立信、沃尔沃、Spotify 这类公司都接受学生做论文。也有少数一年制项目，但计算机方向基本都是两年。"
     "两年制的好处是课程更系统，第二年还能顺便找实习和工作；毕业后可以申请一年的找工作签证。具体课程设置以各校官网为准。"),
    ("申请需要准备哪些材料？",
     "常规材料有：本科成绩单和在读证明（毕业后换成学位证和毕业证）、语言成绩（雅思或托福）、简历、动机信、护照首页，部分学校还要推荐信。"
     "成绩单和证书需要学校盖章的英文版，有的国家还要求公证。动机信要写清楚为什么选这个项目、你的相关经历和以后的规划，"
     "这是材料里最能体现个人差异的部分。建议提前两三个月开始准备，尤其是推荐信要给老师留足时间。各校的具体要求在官网的申请页面都能查到。"),
    ("计算机专业对编程能力有要求吗？",
     "大多数计算机硕士项目没有统一的编程考试，但会看你本科的相关课程，比如数据结构、算法、操作系统、编程语言这些，学分是否够。"
     "有的项目会写明需要多少 ECTS 的计算机课程和数学课程，不够的话即使成绩好也可能被拒。另外 GitHub 项目、实习经历、竞赛奖项在动机信和简历里"
     "能加分。如果你本科是跨专业的，建议先对照目标项目的先修课要求，把缺的课程列出来，看看能不能通过辅修或网课补上。"),
    ("学费大概多少？",
     "北欧几个国家对欧盟以外的学生普遍收学费。瑞典每年大约 12 万到 20 万人民币，丹麦和芬兰也在这个区间，荷兰略低一些。"
     "德国的公立大学大多不收学费，只交每学期几千人民币的注册费，但巴登-符腾堡州例外，每学期要交约 1500 欧元。"
     "生活费另算，北欧大城市每月大约 7000 到 9000 人民币。很多学校有针对优秀国际学生的学费减免奖学金，申请时可以一起勾选。"),
    ("奖学金好申请吗？",
     "奖学金竞争都比较激烈。瑞典有瑞典学会奖学金和各校自己的学费减免，荷兰有橙色郁金香奖学金和学校奖学金，芬兰的学校奖学金覆盖面相对大一些。"
     "评选主要看本科成绩、相关经历和动机信，通常和入学申请同时提交，截止日期也一样，甚至更早。成绩排名靠前、有科研或项目经历的同学机会更大。"
     "建议把奖学金当作加分项，不要把全部预算都押在上面。"),
    ("什么时候开始申请比较好？",
     "以秋季入学为例，瑞典的统一申请平台一般在前一年 10 月中旬开放，次年 1 月中旬截止；芬兰在 1 月；荷兰各校时间不同，热门项目 1 月到 4 月；"
     "德国很多学校是 5 月到 7 月截止，春季入学则在前一年的秋天。所以最好提前一年开始准备：先定国家和项目，再准备语言成绩，"
     "秋天准备文书和材料，开放后尽早提交。热门项目滚动录取的，越早越好。"),
    ("动机信应该怎么写？",
     "动机信的核心是回答三个问题：为什么是这个专业、为什么是这所学校的这个项目、为什么是你。开头简短说明申请意向，中间用一两段具体经历"
     "（课程项目、实习、科研）证明你的兴趣和能力，注意写清楚你做了什么、学到了什么，而不是罗列。再写这个项目的哪些课程或研究方向吸引你，"
     "最后讲毕业后的规划。篇幅一般一到两页，语言要具体、真诚，避免套话。我们的文书服务会帮你梳理经历、多轮修改。"),
    ("毕业后能留下来工作吗？",
     "北欧几个国家对毕业生都比较友好。瑞典毕业后可以申请一年的找工作居留，找到工作后转工作签证；芬兰和丹麦也有类似的找工作期。"
     "荷兰有专门的高技术移民和应届毕业生的“求职年”签证，门槛相对低。计算机方向的就业机会主要集中在斯德哥尔摩、哥本哈根、赫尔辛基、"
     "阿姆斯特丹这些城市。英语工作环境普遍，但学一些当地语言会加分。具体政策每年会调整，以各国移民局官网为准。"),
    ("你们的服务包括哪些？",
     "我们主要做两类服务：选校定位和文书辅导。选校全案包括背景评估、国家和项目对比、定校清单和申请时间规划；文书包括动机信、简历和推荐信的"
     "梳理与多轮修改，可以按项目数量选套餐。另外还有全程陪跑服务，从选校、文书到递交、面试和签证一路跟进。价格可以用报价工具算，"
     "也可以先约一次免费的咨询，让顾问根据你的背景给建议。"),
    ("需要面试吗？",
     "大部分北欧和荷兰的硕士项目不需要面试，主要看材料。少数项目会有线上面试或者编程测试，比如一些竞争激烈的计算机或数据科学方向。"
     "如果有面试，一般是 20 到 30 分钟的视频，问你的项目经历、为什么申请、对课程的了解，也可能问几个基础的技术问题。"
     "提前准备好用英语讲清楚自己的两三个项目，基本就够了。"),
    ("德国申请要 APS 吗？",
     "中国大陆的学生申请德国大学，基本都需要先通过 APS 审核，拿到 APS 证书后才能递交申请和办签证。APS 主要审核学历真实性和成绩，"
     "流程是线上注册、提交材料、缴费，然后等审核，有的情况还有面谈。整个周期大约一到三个月，旺季可能更长，所以要尽早办理。"
     "瑞典、荷兰、芬兰、丹麦不需要 APS。"),
    ("宿舍好找吗？",
     "北欧的学生宿舍比较紧张，尤其是斯德哥尔摩、阿姆斯特丹、哥本哈根。很多学校会给国际新生保证第一年宿舍，但需要拿到录取后尽快申请，"
     "有的要交押金。没有保证宿舍的城市就要靠学生住房基金会排队或者租私人房源，建议提前几个月开始找，警惕要求先打款的租房骗局。"
     "价格大约每月 3000 到 6000 人民币，看城市和房型。"),
]

# 闲聊陈述：带“我”、不是提问，会被存成笔记，用来考检索能不能在一堆相似的话里找对（含干扰项）
NOISE: List[str] = [
    "我觉得北欧冬天太冷了，有点担心适应不了。",
    "我朋友去年去了德国读书，说那边生活挺方便的。",
    "我平时喜欢打篮球，周末一般去健身房。",
    "我爸妈希望我毕业以后回国工作。",
    "我对人工智能方向挺感兴趣的，看过一些机器学习的网课。",
    "我室友在准备考公，我们宿舍好几个人在考研。",
    "我之前看过一个瑞典留学的视频，感觉那边很不错。",
    "我英语口语一般，听力还行。",
    "我们学校今年出国的人挺多的，大概有三十个。",
    "我高中同学在英国读本科，说学费特别贵。",
    "我有点担心一个人在国外会孤单。",
    "我暑假打算先去考个驾照。",
]

ACK = "好的，我记下了。你可以继续问我关于选校、申请流程或者服务的问题。"


def _session(fact_turns: List[Tuple[int, str]], n_turns: int, offset: int) -> List[Tuple[str, str]]:
    """拼一个会话：fact_turns 是 (第几轮, 用户说的话)，其余轮用填充问答。"""
    facts = dict(fact_turns)
    turns: List[Tuple[str, str]] = []
    k = offset
    for i in range(1, n_turns + 1):
        if i in facts:
            turns.append((facts[i], ACK))
        else:
            turns.append(FILLERS[k % len(FILLERS)])
            k += 1
    return turns


def build_scenarios() -> List[Dict[str, Any]]:
    S: List[Dict[str, Any]] = []

    def add(sid, cat, sessions, question, expect, reject=(), probe_conv=None):
        S.append({"id": sid, "category": cat, "sessions": sessions,
                  "probe": {"conv": probe_conv or sessions[-1]["conv"], "question": question,
                            "expect": list(expect), "reject": list(reject)}})

    # early_recall：第 1 轮的事实，5–7 轮后问
    add("E1", "early_recall", [{"conv": "a", "turns": _session([(1, "你好，我本科是西安电子科技大学通信工程专业的，想了解出国读研。")], 5, 0)}],
        "我本科学的是什么专业？", ["通信"])
    add("E2", "early_recall", [{"conv": "a", "turns": _session([(1, "我的雅思是 6.5，小分最低 6，不知道够不够。")], 6, 1)}],
        "我的雅思总分是多少？", ["6.5"])
    add("E3", "early_recall", [{"conv": "a", "turns": _session([(1, "我打算 2027 年秋季入学。")], 7, 2)}],
        "我计划哪一年入学？", ["2027"])
    add("E4", "early_recall", [{"conv": "a", "turns": _session([(1, "家里给的预算是每年 15 万人民币左右。")], 6, 3)}],
        "我每年的预算大概多少？", ["15"])
    add("E5", "early_recall", [{"conv": "a", "turns": _session([(1, "我现在大四，在一家互联网公司做后端开发实习。")], 7, 4)}],
        "我现在在做什么方向的实习？", ["后端"])
    add("E6", "early_recall", [{"conv": "a", "turns": _session([(1, "我最想去的是芬兰，喜欢那边安静的环境。")], 5, 5)}],
        "我最想去哪个国家？", ["芬兰"])
    # post_compress：已经滑出最近对话之后再问
    add("P1", "post_compress", [{"conv": "a", "turns": _session([(1, "我本科读的是自动化专业。")], 9, 6)}],
        "我本科是什么专业？", ["自动化"])
    add("P2", "post_compress", [{"conv": "a", "turns": _session([(1, "我的 GPA 是 3.4，满分 4 分。")], 10, 7)}],
        "我的 GPA 是多少？", ["3.4"])
    add("P3", "post_compress", [{"conv": "a", "turns": _session([(1, "我的目标国家是丹麦，别的国家先不考虑。")], 11, 8)}],
        "我的目标国家是哪个？", ["丹麦"])
    add("P4", "post_compress", [{"conv": "a", "turns": _session([(1, "整个留学的总预算大概 40 万。")], 10, 9)}],
        "我的总预算是多少？", ["40"])
    # update：中途改主意，问现在
    add("U1", "update", [{"conv": "a", "turns": _session([(1, "我想去荷兰读计算机硕士。"), (4, "我和家里商量了，决定不去荷兰了，改申瑞典。")], 6, 10)}],
        "我现在的目标国家是哪个？", ["瑞典"], ["荷兰"])
    add("U2", "update", [{"conv": "a", "turns": _session([(1, "我想 2027 年春季入学。"), (5, "入学时间改成 2027 年秋季吧，春季准备不过来。")], 9, 11)}],
        "我现在打算哪个季节入学？", ["秋"], ["春"])
    add("U3", "update", [{"conv": "a", "turns": _session([(1, "我雅思现在是 6.0。"), (3, "上周重考了，雅思出分 7.0！")], 10, 0)}],
        "我现在的雅思成绩是多少？", ["7"], ["6.0"])
    add("U4", "update", [{"conv": "a", "turns": _session([(2, "预算大概每年 20 万。"), (6, "家里情况有变化，预算降到每年 12 万了。")], 8, 1)}],
        "我现在每年的预算是多少？", ["12"], ["20"])
    # history：问最开始的值
    add("H1", "history", [{"conv": "a", "turns": _session([(1, "我想去荷兰读计算机硕士。"), (4, "我和家里商量了，决定不去荷兰了，改申瑞典。")], 8, 2)}],
        "我最开始想去哪个国家？", ["荷兰"])
    add("H2", "history", [{"conv": "a", "turns": _session([(1, "我雅思第一次考了 6.0。"), (3, "上周重考了，雅思出分 7.0！")], 9, 3)}],
        "我第一次考雅思是多少分？", ["6"], ["7"])
    # cross_session：新会话里问
    add("C1", "cross_session", [
        {"conv": "a", "turns": _session([(1, "我想申请德国的计算机硕士，2028 年冬季学期入学。")], 6, 4)},
        {"conv": "b", "turns": []}],
        "我想去哪个国家读书？", ["德国"], probe_conv="b")
    add("C2", "cross_session", [
        {"conv": "a", "turns": _session([(1, "我本科是数学专业，GPA 3.6。")], 10, 5)},
        {"conv": "b", "turns": []}],
        "我本科是什么专业？", ["数学"], probe_conv="b")
    # abstention：没说过的事
    add("A1", "abstention", [{"conv": "a", "turns": _session([(1, "我想去瑞典读书。")], 6, 6)}],
        "我的预算是多少？", ["未知"])
    add("A2", "abstention", [{"conv": "a", "turns": _session([(1, "我雅思考了 6.5。")], 7, 7)}],
        "我的托福考了多少分？", ["未知"], ["6.5"])
    for sc in S:
        sc["set"] = "basic"

    # ── 难题：更接近真实使用（多个事实、跨多个会话、长消息） ──
    hard: List[Dict[str, Any]] = []
    five = [(1, "我本科是华中科技大学的电子信息工程专业。"), (2, "我的 GPA 是 3.5。"), (3, "想去荷兰，2027 年秋季入学。"),
            (4, "预算是每年 18 万。"), (5, "雅思 6.5，小分都过 6 了。")]

    def addh(sid, cat, sessions, question, expect, reject=(), probe_conv=None):
        add(sid, cat, sessions, question, expect, reject, probe_conv)
        S[-1]["set"] = "hard"
        hard.append(S[-1])

    addh("M1", "multi_fact", [{"conv": "a", "turns": _session(five, 17, 0)}], "我本科学的是什么专业？", ["电子信息"])
    addh("M2", "multi_fact", [{"conv": "a", "turns": _session(five, 17, 3)}], "我的 GPA 是多少？", ["3.5"])
    addh("M3", "multi_fact", [{"conv": "a", "turns": _session(five, 17, 6)}], "我每年的预算是多少？", ["18"])
    addh("M4", "multi_fact", [{"conv": "a", "turns": _session(five, 17, 9)}], "我的托福考了多少分？", ["未知"], ["6.5"])
    three = [{"conv": "a", "turns": _session([(1, "我想去荷兰读计算机硕士，本科是软件工程。")], 6, 1)},
             {"conv": "b", "turns": _session([(2, "跟你说一下，我已经决定不去荷兰了，改申瑞典。")], 6, 5)},
             {"conv": "c", "turns": _session([], 2, 8)}]
    addh("X1", "cross_update", three, "我现在的目标国家是哪个？", ["瑞典"], ["荷兰"], probe_conv="c")
    addh("X2", "cross_update", three, "我最开始想去哪个国家？", ["荷兰"], probe_conv="c")
    addh("X3", "cross_update", three, "我本科是什么专业？", ["软件"], probe_conv="c")
    long_msg = ("我先介绍一下情况：我在一所双非院校读计算机，大一大二成绩一般，大三开始认真学，做了两个课程项目，一个是基于 Spring Boot 的"
                "选课系统，另一个是用 PyTorch 做的图像分类，还参加过一次数学建模比赛拿了省二等奖。现在在准备语言考试，同时在纠结是直接工作"
                "还是出国读研，家里比较支持出国，但也担心毕业后不好找工作，所以想先多了解一些情况再做决定。对了，我的托福刚考出来是 98 分。")
    addh("L1", "long_message", [{"conv": "a", "turns": _session([(1, long_msg)], 6, 2)}], "我的托福是多少分？", ["98"])
    addh("L2", "long_message", [{"conv": "a", "turns": _session([(1, long_msg)], 11, 4)}], "我的托福是多少分？", ["98"])
    addh("T1", "long_conversation", [{"conv": "a", "turns": _session([(2, "补充一下，我是 2025 年本科毕业的，已经工作一年了。")], 25, 7)}],
         "我是哪一年本科毕业的？", ["2025"])

    # ── 压力题：闲聊陈述句也会存成笔记（含干扰项），多个会话、反复修改、相对时间、多跳。和 basic / hard 分开跑 ──
    def chat(facts: List[Tuple[int, str]], n_turns: int, offset: int) -> List[Tuple[str, str]]:
        """事实 + 闲聊陈述 + 常见提问交替，比 _session 更像真实用户。"""
        facts_at = dict(facts)
        turns: List[Tuple[str, str]] = []
        for i in range(1, n_turns + 1):
            if i in facts_at:
                turns.append((facts_at[i], ACK))
            elif i % 2:
                turns.append((NOISE[(offset + i) % len(NOISE)], ACK))
            else:
                turns.append(FILLERS[(offset + i) % len(FILLERS)])
        return turns

    def adds(sid, cat, sessions, question, expect, reject=()):
        probe = f"p{len(sessions)}"
        add(sid, cat, sessions + [{"conv": probe, "turns": []}], question, expect, reject, probe_conv=probe)
        S[-1]["set"] = "stress"

    adds("S1", "many_sessions", [
        {"conv": "a", "turns": chat([(2, "我本科是华中科技大学的，读电子信息工程。"), (6, "GPA 3.5。")], 12, 0)},
        {"conv": "b", "turns": chat([], 10, 3)}, {"conv": "c", "turns": chat([], 10, 6)}],
        "我本科是哪所学校的？", ["华中科技"])
    budget = [{"conv": "a", "turns": chat([(2, "家里给的预算是每年 20 万。")], 8, 1)},
              {"conv": "b", "turns": chat([(3, "家里生意不太好，预算降到每年 15 万了。")], 8, 4)},
              {"conv": "c", "turns": chat([(2, "好消息，我拿到一笔奖学金，加上家里的钱，每年能到 25 万了。")], 8, 7)}]
    adds("S2", "repeated_update", budget, "我现在每年的预算是多少？", ["25"], ["20 万", "15 万"])
    adds("S3", "repeated_history", budget, "我最开始说的每年预算是多少？", ["20"], ["25"])
    adds("S4", "temporal", [
        {"conv": "a", "turns": chat([(3, "我去年夏天第一次考雅思，考了 6.0。")], 10, 2)},
        {"conv": "b", "turns": chat([], 8, 5)}],
        "我第一次考雅思是哪一年？", ["2025"])
    adds("S5", "multi_hop", [
        {"conv": "a", "turns": chat([(3, "我女朋友在哥本哈根工作，已经两年了。")], 10, 3)},
        {"conv": "b", "turns": chat([(4, "我最想的是读书的时候能和女朋友在同一个城市。")], 8, 6)}],
        "我想去哪个城市读书？", ["哥本哈根"])
    adds("S6", "distractor", [
        {"conv": "a", "turns": chat([(3, "我表姐在荷兰读的硕士，现在在阿姆斯特丹工作。"), (7, "不过我自己想去芬兰，喜欢安静一点的地方。")], 12, 1)},
        {"conv": "b", "turns": chat([], 8, 9)}],
        "我自己想去哪个国家？", ["芬兰"], ["荷兰"])
    adds("S7", "abstention", [
        {"conv": "a", "turns": chat([(2, "我 GPA 3.3，雅思 6.5，托福没考。"), (6, "预算每年 18 万。")], 12, 4)},
        {"conv": "b", "turns": chat([], 8, 2)}],
        "我的 GRE 考了多少分？", ["未知"])
    adds("S8", "long_history", [
        {"conv": "a", "turns": chat([(3, "我的 GPA 是 3.7，专业排名前 10%。")], 40, 5)}],
        "我的 GPA 是多少？", ["3.7"])
    adds("S9", "cross_update", [
        {"conv": "a", "turns": chat([(2, "我想去荷兰读数据科学。")], 8, 0)},
        {"conv": "b", "turns": chat([], 8, 3)},
        {"conv": "c", "turns": chat([(5, "跟你说一下，荷兰的项目太贵了，我改申瑞典了。")], 8, 6)},
        {"conv": "d", "turns": chat([], 6, 9)}],
        "我现在的目标国家是哪个？", ["瑞典"], ["荷兰"])
    adds("S10", "detail_recall", [
        {"conv": "a", "turns": chat([(2, "我参加过全国大学生数学建模竞赛，拿了省级二等奖。")], 10, 7)},
        {"conv": "b", "turns": chat([], 10, 1)}],
        "我参加数学建模竞赛拿了什么奖？", ["二等奖"])
    return S


READER_SYSTEM = (
    "你是留学咨询助手。下面的[背景信息]是系统提供的关于这位用户的记忆。用户会问关于他自己的问题，请只根据[背景信息]回答。"
    "只输出答案本身（几个字到一个短语），不要解释。如果背景信息里没有，就输出“未知”。"
    "同一件事前后说法不同时，以最新的为准；只有用户问的是以前的情况时才回答以前的。"
)
ABSTAIN_WORDS = ("未知", "不知道", "没有提到", "未提及", "没有相关")


def score(answer: str, expect: List[str], reject: List[str]) -> bool:
    a = (answer or "").strip()
    if expect == ["未知"]:
        ok = any(w in a for w in ABSTAIN_WORDS)
    else:
        ok = all(e in a for e in expect)
    return ok and not any(r in a for r in reject)


class Counter:
    def __init__(self) -> None:
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def wrap(self, client: Any) -> None:
        create = client.messages.create

        async def counted(*args: Any, **kwargs: Any) -> Any:
            resp = await create(*args, **kwargs)
            self.calls += 1
            u = getattr(resp, "usage", None)
            self.tokens_in += int(getattr(u, "input_tokens", 0) or 0) + int(getattr(u, "cache_read_input_tokens", 0) or 0)
            self.tokens_out += int(getattr(u, "output_tokens", 0) or 0)
            return resp

        client.messages.create = counted


async def run_once(run_idx: int, label: str, scenarios: List[Dict[str, Any]], concurrency: int) -> Dict[str, Any]:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    model = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-flash")
    mgr = MemoryManager(redis_url="redis://unused:1/0", chroma_host="",
                        chroma_path=tempfile.mkdtemp(prefix=f"mem-{label}-{run_idx}-"),
                        api_key=api_key, base_url=base_url, model=model)
    mgr._redis = FakeRedis()
    maint = Counter()
    maint.wrap(mgr._client)
    reader = make_client(api_key, base_url, "eval")
    sem = asyncio.Semaphore(concurrency)
    stamp = f"{label}-r{run_idx}-{int(time.time())}"

    async def one(sc: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            uid = f"{stamp}-{sc['id']}"
            for sess in sc["sessions"]:
                conv = f"{uid}-{sess['conv']}"
                for user_msg, assistant_msg in sess["turns"]:
                    await mgr.add_message(uid, conv, MsgRole.USER, user_msg)
                    await mgr.add_message(uid, conv, MsgRole.ASSISTANT, assistant_msg)
                    await mgr.after_turn(uid, conv)          # 对话接口每轮回复后都会调用
                if sess["turns"]:
                    await mgr.enrich_notes(uid)              # 会话之间用户停下来了：线上空闲 IDLE_SECONDS 后补全
            await mgr.wait_background()
            probe = sc["probe"]
            ctx = await mgr.get_context(uid, f"{uid}-{probe['conv']}", query=probe["question"])
            ctx_text = ctx.to_prompt_text()
            resp = await reader.messages.create(
                model=model, max_tokens=64, temperature=0.0, system=READER_SYSTEM,
                messages=[{"role": "user", "content": f"[背景信息]\n{ctx_text or '（无）'}\n\n问题：{probe['question']}"}],
                **NO_THINKING_KWARGS,
            )
            answer = extract_text_content(resp.content).strip()
            return {"id": sc["id"], "category": sc["category"], "question": probe["question"],
                    "answer": answer, "correct": score(answer, probe["expect"], probe["reject"]),
                    "context_chars": len(ctx_text), "context": ctx_text}

    results = await asyncio.gather(*(one(sc) for sc in scenarios))
    return {"run": run_idx, "results": results, "maintenance_calls": maint.calls,
            "maintenance_tokens_in": maint.tokens_in, "maintenance_tokens_out": maint.tokens_out}


def summarize(label: str, runs: List[Dict[str, Any]], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    cats = sorted({s["category"] for s in scenarios}, key=lambda c: [s["category"] for s in scenarios].index(c))
    per_cat: Dict[str, Dict[str, Any]] = {}
    for c in cats:
        ids = [s["id"] for s in scenarios if s["category"] == c]
        accs = [sum(r["correct"] for r in run["results"] if r["id"] in ids) / len(ids) for run in runs]
        pass_all = sum(all(next(r for r in run["results"] if r["id"] == i)["correct"] for run in runs) for i in ids) / len(ids)
        per_cat[c] = {"n": len(ids), "acc_mean": sum(accs) / len(accs), "acc_runs": accs, "pass_k": pass_all}
    all_ids = [s["id"] for s in scenarios]
    overall = [sum(r["correct"] for r in run["results"]) / len(all_ids) for run in runs]
    pass_k = sum(all(next(r for r in run["results"] if r["id"] == i)["correct"] for run in runs) for i in all_ids) / len(all_ids)
    ctx = [r["context_chars"] for run in runs for r in run["results"]]
    return {
        "label": label, "runs": len(runs), "scenarios": len(all_ids),
        "overall_acc_mean": sum(overall) / len(overall), "overall_acc_runs": overall, "pass_k": pass_k,
        "per_category": per_cat,
        "maintenance_calls_per_run": sum(r["maintenance_calls"] for r in runs) / len(runs),
        "maintenance_tokens_per_run": sum(r["maintenance_tokens_in"] + r["maintenance_tokens_out"] for r in runs) / len(runs),
        "avg_context_chars": sum(ctx) / len(ctx),
    }


def write_report(summary: Dict[str, Any], runs: List[Dict[str, Any]]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = REPORT_DIR / f"memory_{summary['label']}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    k = summary["runs"]
    lines = [f"# 多轮记忆评测：{summary['label']}", "",
             f"- 场景 {summary['scenarios']} 个，每个跑 {k} 次",
             f"- 总体准确率（{k} 次平均）：**{summary['overall_acc_mean']:.1%}**；各次：{', '.join(f'{x:.0%}' for x in summary['overall_acc_runs'])}",
             f"- pass^{k}（同一场景 {k} 次全对的比例）：**{summary['pass_k']:.1%}**",
             f"- 每次运行的记忆维护调用（笔记补全）：{summary['maintenance_calls_per_run']:.0f} 次，"
             f"约 {summary['maintenance_tokens_per_run']:.0f} token",
             f"- 提问时记忆文本平均长度：{summary['avg_context_chars']:.0f} 字", "",
             "| 类别 | 场景数 | 平均准确率 | pass^k |", "|---|---|---|---|"]
    for c, v in summary["per_category"].items():
        lines.append(f"| {c} | {v['n']} | {v['acc_mean']:.0%} | {v['pass_k']:.0%} |")
    lines += ["", "## 每个场景（第 1 次运行）", "", "| 场景 | 问题 | 回答 | 对错 |", "|---|---|---|---|"]
    for r in runs[0]["results"]:
        lines.append(f"| {r['id']} | {r['question']} | {r['answer'].replace('|', '/')} | {'✅' if r['correct'] else '❌'} |")
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base.with_suffix(".md")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", default="", help="只跑这些场景，逗号分隔，如 E1,U1")
    ap.add_argument("--set", default="all", choices=["basic", "hard", "all", "stress"], help="basic 基础题 / hard 难题 / all 前两者 / stress 压力题（单独跑）")
    args = ap.parse_args()
    scenarios = [sc for sc in build_scenarios() if sc["set"] == args.set or (args.set == "all" and sc["set"] != "stress")]
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    runs = [await run_once(i + 1, args.label, scenarios, args.concurrency) for i in range(args.runs)]
    summary = summarize(args.label, runs, scenarios)
    path = write_report(summary, runs)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_category"}, ensure_ascii=False))
    for c, v in summary["per_category"].items():
        print(f"  {c:<14} acc {v['acc_mean']:.0%}  pass^k {v['pass_k']:.0%}")
    print("report:", path)


if __name__ == "__main__":
    asyncio.run(main())
