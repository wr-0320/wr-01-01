"""Generate the Chinese master's thesis Word document from frozen artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "thesis"
TITLE = "部分可观察对抗环境中的主动反欺骗决策研究"
SUBTITLE = "——基于双层信念、反事实推理与自适应承诺的 RIDe 框架"


def set_run_font(run, east_asia, latin=None, size=None):
    run.font.name = latin or east_asia
    if size is not None:
        run.font.size = Pt(size)
    properties = run._element.get_or_add_rPr()
    fonts = properties.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        properties.insert(0, fonts)
    fonts.set(qn("w:eastAsia"), east_asia)
    fonts.set(qn("w:ascii"), latin or east_asia)
    fonts.set(qn("w:hAnsi"), latin or east_asia)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_cell_text(cell, text, size=8.5, bold=False):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(str(text))
    run.bold = bold
    set_run_font(run, "宋体", "Times New Roman", size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def shade_cell(cell, fill):
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    element = OxmlElement("w:tblHeader")
    element.set(qn("w:val"), "true")
    tr_pr.append(element)


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, end])


class Thesis:
    def __init__(self):
        self.doc = Document()
        self.figure_number = 0
        self.table_number = 0
        self._configure()

    def _configure(self):
        section = self.doc.sections[0]
        section.page_height = Cm(29.7)
        section.page_width = Cm(21.0)
        section.top_margin = Cm(2.6)
        section.bottom_margin = Cm(2.4)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)
        section.header_distance = Cm(1.3)
        section.footer_distance = Cm(1.3)

        styles = self.doc.styles
        normal = styles["Normal"]
        normal.font.name = "Times New Roman"
        normal.font.size = Pt(12)
        normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        normal.paragraph_format.first_line_indent = Cm(0.85)
        normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        normal.paragraph_format.space_after = Pt(0)
        normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        heading_specs = {
            "Title": ("黑体", 22, True, WD_ALIGN_PARAGRAPH.CENTER),
            "Heading 1": ("黑体", 18, True, WD_ALIGN_PARAGRAPH.CENTER),
            "Heading 2": ("黑体", 15, True, WD_ALIGN_PARAGRAPH.LEFT),
            "Heading 3": ("黑体", 13, True, WD_ALIGN_PARAGRAPH.LEFT),
        }
        for name, (font, size, bold, alignment) in heading_specs.items():
            style = styles[name]
            style.font.name = font
            style.font.size = Pt(size)
            style.font.bold = bold
            style._element.rPr.rFonts.set(qn("w:eastAsia"), font)
            style.paragraph_format.alignment = alignment
            style.paragraph_format.space_before = Pt(12)
            style.paragraph_format.space_after = Pt(12)
        styles["Heading 1"].paragraph_format.page_break_before = True

        if "公式" not in styles:
            equation = styles.add_style("公式", WD_STYLE_TYPE.PARAGRAPH)
            equation.font.name = "Cambria Math"
            equation.font.size = Pt(11)
            equation.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            equation.paragraph_format.space_before = Pt(6)
            equation.paragraph_format.space_after = Pt(6)
        if "图表标题" not in styles:
            caption = styles.add_style("图表标题", WD_STYLE_TYPE.PARAGRAPH)
            caption.font.name = "宋体"
            caption.font.size = Pt(10.5)
            caption._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
            caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption.paragraph_format.space_after = Pt(8)

        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = header.add_run(TITLE)
        set_run_font(run, "宋体", "Times New Roman", 9)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(footer, "PAGE")

        settings = self.doc.settings.element
        update = OxmlElement("w:updateFields")
        update.set(qn("w:val"), "true")
        settings.append(update)

    def page_break(self):
        self.doc.add_page_break()

    def chapter(self, title):
        self.doc.add_heading(title, level=1)

    def section(self, title):
        self.doc.add_heading(title, level=2)

    def subsection(self, title):
        self.doc.add_heading(title, level=3)

    def p(self, text, bold_prefix=None):
        paragraph = self.doc.add_paragraph()
        if bold_prefix and text.startswith(bold_prefix):
            run = paragraph.add_run(bold_prefix)
            run.bold = True
            paragraph.add_run(text[len(bold_prefix):])
        else:
            paragraph.add_run(text)
        return paragraph

    def equation(self, text):
        self.doc.add_paragraph(text, style="公式")

    def bullet(self, text, level=0):
        style = "List Bullet" if level == 0 else "List Bullet 2"
        paragraph = self.doc.add_paragraph(text, style=style)
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.35
        return paragraph

    def algorithm(self, title, lines):
        self.table_number += 1
        caption = self.doc.add_paragraph(f"算法 {self.table_number}  {title}", style="图表标题")
        caption.paragraph_format.keep_with_next = True
        table = self.doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Table Grid"
        cell = table.cell(0, 0)
        cell.text = ""
        for index, line in enumerate(lines):
            paragraph = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
            paragraph.paragraph_format.first_line_indent = Cm(0)
            paragraph.paragraph_format.space_after = Pt(1)
            run = paragraph.add_run(line)
            set_run_font(run, "等线", "Consolas", 9)
        return table

    def table(self, title, headers, rows, widths=None, font_size=8.5):
        self.table_number += 1
        self.doc.add_paragraph(f"表 {self.table_number}  {title}", style="图表标题")
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        set_repeat_table_header(table.rows[0])
        for idx, header in enumerate(headers):
            set_cell_text(table.rows[0].cells[idx], header, font_size, True)
            shade_cell(table.rows[0].cells[idx], "D9EAF7")
        for row in rows:
            cells = table.add_row().cells
            for idx, value in enumerate(row):
                set_cell_text(cells[idx], value, font_size)
                if widths and idx < len(widths):
                    cells[idx].width = Cm(widths[idx])
        self.doc.add_paragraph()
        return table

    def figure(self, path, title, width=6.1):
        path = Path(path)
        if not path.exists():
            return
        self.figure_number += 1
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.add_run().add_picture(str(path), width=Inches(width))
        self.doc.add_paragraph(f"图 {self.figure_number}  {title}", style="图表标题")

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(path)


def add_cover(t: Thesis):
    for _ in range(3):
        t.doc.add_paragraph()
    p = t.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("硕 士 学 位 论 文")
    run.bold = True
    set_run_font(run, "黑体", "Arial", 26)
    for _ in range(3):
        t.doc.add_paragraph()
    p = t.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(TITLE)
    run.bold = True
    set_run_font(run, "黑体", "Arial", 22)
    p = t.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(SUBTITLE)
    set_run_font(run, "黑体", "Arial", 16)
    for _ in range(5):
        t.doc.add_paragraph()
    info = [
        ("学位申请人", "（请填写）"),
        ("指导教师", "（请填写）"),
        ("学科专业", "计算机科学与技术（请按学校要求修改）"),
        ("培养单位", "（请填写）"),
        ("完成日期", "2026 年 10 月"),
    ]
    table = t.doc.add_table(rows=len(info), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(info):
        set_cell_text(table.cell(i, 0), row[0], 14, True)
        set_cell_text(table.cell(i, 1), row[1], 14)
    t.page_break()
    t.chapter("学位论文原创性声明与使用授权说明")
    t.p("本人郑重声明：所呈交的学位论文是在导师指导下独立完成的研究成果。除文中已经注明引用的内容外，本论文不包含他人已经发表或撰写的研究成果。对本研究做出贡献的个人和集体，均已在文中以明确方式标明。")
    t.p("本人同意学校按照有关规定保存、使用本学位论文，并授权学校将论文的全部或部分内容编入有关数据库进行检索。涉及保密内容时，应按学校规定另行办理。")
    t.p("作者签名：________________    日期：________________")
    t.p("导师签名：________________    日期：________________")


def add_abstracts(t: Thesis):
    t.chapter("摘  要")
    paragraphs = [
        "在部分可观察的对抗环境中，智能体的运动轨迹既是实现任务的控制序列，也是向观察者传递意图的行为信号。具备策略性的攻击者可以利用这一双重属性，通过先向错误目标移动、延迟反转、多次切换或根据防守者行为调整诱导方式，使防守者形成错误信念。现有意图识别方法往往将目标预测直接等同于防守决策，因而可能在分类指标较高时仍作出错误且不可逆的拦截承诺。本文研究双目标部分可观察环境中的主动反欺骗决策问题，重点回答三个问题：如何区分目标不确定性与欺骗不确定性，如何利用反事实诚实行为识别策略偏离，以及如何在观察成本与错误承诺风险之间选择行动。",
        "本文将反欺骗建模为一个由行为观察、隐状态推断、信息可靠性评估和不可逆承诺共同构成的序贯决策问题。基础场景为双智能体 7×7 GridWorld：攻击者知道真实目标 A/B，防守者只能获得局部可见轨迹及时间信息。环境、信念估计器和行动控制器相互解耦，使规则推断、校准模型、学习模型与传统规划方法能够在一致的信息边界下比较。攻击集合覆盖诚实、随机、固定时序诱导、延迟反转、绕行、多次切换、随机欺骗以及根据防守行为调整策略的自适应攻击。",
        "针对目标推断与防御行动相互混淆的问题，本文提出 RIDe 主动反欺骗框架。其推理层采用共享 GRU 编码器和两个预测头，分别估计目标信念与欺骗概率，并使用诚实轨迹原型计算当前行为在诚实解释和欺骗解释下的反事实似然比；校准模块进一步输出可用于决策的置信度与信念熵。其决策层综合目标后验、欺骗证据、不确定性、累计观察成本和错误承诺风险，在 observe、track、commit_A 与 commit_B 之间选择动作。目标判断与动作控制因而可以分别训练、校准和审计，且承诺一旦发生便不可逆。理论分析给出可观察别名条件下历史分类器的 Bayes 误差下界，并说明检测性能与拦截性能之间不存在直接等价关系。",
        "实验揭示了三类关键现象。其一，PPO 欺骗者可能出现停留坍塌、边界碰撞、奖励投机和目标不对称，修改奖励比例不能自动恢复任务完成。其二，平衡可辨识性数据中存在 32 个不可约错误，对应 93.75% 的有限数据经验上限；Bayesian 和有限时域 POMDP 在固定时序欺骗下可取得 1.0 的检测 AUROC，却得到 0% 拦截率，直接展示检测—决策鸿沟。其三，在冻结模型的泛化诊断中，决策时 belief error 从已见攻击的 0.0125 增至未见攻击的 0.6465，拦截率从 1.000 降至 0.288；712 个失败回合中有 433 个满足严格可观察别名规则，表明信息不足、反事实假设失配和攻击分布偏移共同限制了历史信息驱动的方法。",
        "多环境完整基准在 GridWorld、Forked Routes、PettingZoo deception 和 CyberDeceptionEnv 上比较 Rule Active、Bayesian、POMCP、Active Inference、MAPPO 与 RIDe。实验使用 5 个随机种子，覆盖 scripted、adaptive 和 unseen family，每个方法—环境—攻击条件包含 500 个回合，共计 36,000 个评估回合。RIDe 在 GridWorld、Forked Routes 和 Cyber 环境的跨攻击族拦截率分别为 82.3%、56.9% 和 98.0%，在 PettingZoo 环境为 52.3%；Rule Active 在 PettingZoo 达到 79.2%。MAPPO 在总计约 500,000 timesteps 后仍收敛到 50% 拦截率和零观察成本的即时承诺策略。结果说明，联合欺骗推理、不确定性估计和主动决策能够在若干观测结构下改善防御，但其优势受环境语义与行为假设约束。",
        "本文的主要贡献包括：将策略性反欺骗重新定义为推断—决策联合分析问题；提出联合目标信念、欺骗信念、反事实证据与自适应承诺的 RIDe 框架；从理论与实证两方面刻画仅依赖可见历史时的可辨识性边界；建立覆盖多攻击族、多环境和多类决策基线的统一评估体系，并以 36,000 个评估回合明确报告方法优势、失败机制与适用条件。",
    ]
    for p in paragraphs:
        t.p(p)
    p = t.doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    r = p.add_run("关键词：")
    r.bold = True
    p.add_run("部分可观察；策略性欺骗；意图识别；反事实推理；主动感知；信念规划；多智能体强化学习")

    t.chapter("ABSTRACT")
    english = [
        "In partially observable adversarial environments, a trajectory is both a control sequence for accomplishing a task and a behavioral signal from which an observer infers intent. A strategic attacker can exploit this dual role by moving toward a false target, delaying reversal, switching repeatedly, or adapting its cues to the defender's sensing and commitment behavior. A defender that directly converts a goal prediction into an irreversible interception decision may therefore fail even when classification metrics appear strong.",
        "This thesis studies active anti-deception decisions with two latent goals. It develops RIDe, a two-level framework that combines a GRU encoder with separate goal and deception heads, a prototype-based counterfactual likelihood ratio, calibrated uncertainty, and an adaptive commitment controller. The controller selects observe, track, commit_A, or commit_B while accounting for information value, sensing cost, and false-commitment risk. Goal inference and action selection are explicitly decoupled.",
        "The empirical study spans reward diagnosis, identifiability analysis, active defense, Bayesian and POMDP baselines, adaptive attackers, failure attribution, and external validation. The full benchmark covers GridWorld, Forked Routes, a PettingZoo deception game, and CyberDeceptionEnv. Six methods are evaluated under scripted, adaptive, and unseen attacks using five random seeds and 500 episodes per method-environment-attack cell, for 36,000 episodes in total.",
        "RIDe obtains macro interception rates of 82.3%, 56.9%, 52.3%, and 98.0% on GridWorld, Forked Routes, PettingZoo deception, and CyberDeceptionEnv, respectively. Its advantage is therefore environment dependent: Rule Active reaches 79.2% on PettingZoo. A 500,000-step MAPPO study converges to an immediate-commit policy with 50% interception, showing that a larger training budget alone does not repair a poor decision structure. Frozen failure attribution further identifies observable aliasing, attacker distribution shift, premature commitment, and counterfactual misspecification as distinct failure mechanisms.",
        "The results support a bounded claim: anti-deception should be treated as a calibrated sequential decision problem rather than trajectory classification alone. Joint beliefs, counterfactual checks, and active commitment can improve defense in several domains, but no history-only method can resolve observationally aliased behaviors without additional information or assumptions.",
    ]
    for p in english:
        t.p(p)
    p = t.doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    r = p.add_run("KEY WORDS: ")
    r.bold = True
    p.add_run("partial observability; strategic deception; intent inference; counterfactual reasoning; active sensing; belief planning; multi-agent reinforcement learning")


def add_front_matter(t: Thesis):
    t.chapter("目  录")
    p = t.doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    add_field(p, 'TOC \\o "1-3" \\h \\z \\u')
    t.p("提示：首次在 Microsoft Word 中打开文档时，请选择目录并执行“更新整个目录”，以刷新页码。")
    t.chapter("主要符号与缩略语")
    rows = [
        ("G", "攻击者的真实目标，G∈{A,B}"),
        ("Z", "行为类型变量，Z=0 表示诚实，Z=1 表示欺骗"),
        ("H_t", "截至时间 t 的防守者可见历史"),
        ("b_t", "目标后验信念 P(G|H_t)"),
        ("q_t", "欺骗后验概率 P(Z=1|H_t)"),
        ("L(H_t)", "欺骗模型与诚实模型的反事实似然比"),
        ("POMDP", "部分可观察马尔可夫决策过程"),
        ("POMCP", "基于蒙特卡洛树搜索的在线 POMDP 规划"),
        ("PPO", "近端策略优化"),
        ("MAPPO", "多智能体近端策略优化"),
        ("AUROC", "受试者工作特征曲线下面积"),
        ("ECE", "期望校准误差"),
        ("RIDe", "本文的反事实推断与自适应决策框架"),
    ]
    t.table("主要符号与缩略语", ["符号", "含义"], rows, [3, 11], 10)


def add_chapter1(t: Thesis):
    t.chapter("第1章  绪论")
    t.section("1.1 研究背景")
    for text in [
        "随着自主智能体进入网络防御、无人系统、机器人协作和关键基础设施等场景，决策者越来越需要根据不完整的行为痕迹推断对手意图。传统轨迹预测通常将运动视为接近目标的物理过程，并假设行动者会以近似最短路径或最大效用方式到达真实目标。然而，在对抗环境中，行为轨迹同时具有信号作用。攻击者可以主动选择一条看似指向错误目标的轨迹，使防守者错误部署资源。",
        "这一问题与一般噪声识别不同。随机噪声不以改变观察者判断为目标，而策略性欺骗依赖观察者的推断规则。攻击者可能知道候选目标、传感器覆盖范围和典型承诺时间，并据此安排诱导阶段与反转阶段。因此，防守策略本身会改变攻击者的最优行为，数据分布也会随防守者行动改变。",
        "部分可观察进一步放大了困难。防守者看不到完整状态，只能间歇获得局部位置、告警或审计线索。同一段可见历史可能对应诚实行为，也可能是尚未反转的欺骗行为。当承诺不可逆时，一次错误判断会直接导致拦截失败；当防守者持续观察而不承诺时，又会承担成本并错过拦截窗口。",
        "由此可见，反欺骗不能只作为二分类问题处理。实用系统至少要同时回答四个问题：当前最可能的目标是什么；观察行为是否偏离诚实预期；当前信念是否足以支持承诺；继续观察带来的信息价值是否超过成本。本文围绕这四个问题建立研究框架。",
    ]:
        t.p(text)
    t.section("1.2 问题提出")
    for text in [
        "本文考虑一个攻击者与防守者组成的双智能体系统。攻击者知道真实目标 A 或 B，防守者知道两个候选目标的位置或语义，但不知道当前真实目标。攻击者可以诚实移动，也可以采用脚本诱导、延迟、绕行、多次切换、随机欺骗和行为自适应等策略。防守者只能使用符合信息边界的可见历史。",
        "当错误承诺被直接用作欺骗奖励时，PPO 攻击者并不会自然产生任务有效的欺骗。策略可能通过停留、反复撞墙或只制造错误信号获取 shaping reward，却不完成真实目标。防守者一侧也存在类似捷径：立即 commit 可以避免观察成本，永不 commit 可以避免错误承诺惩罚。这类现象说明，奖励数值调整无法代替正确的决策结构。",
        "另一方面，检测指标和决策指标可以完全背离。形式化 Bayesian 基线在 Scripted-3 条件下取得 AUROC 1.0，却由于诚实运动模型被诱导而 100% 错误承诺。该现象说明，识别出样本具有欺骗性并不等于知道真实目标，更不等于在正确时间执行正确拦截。",
    ]:
        t.p(text)
    t.section("1.3 研究目标与研究问题")
    t.p("本文的总体目标是建立一个可复现、可诊断、可跨环境比较的主动反欺骗框架，并确定它的有效范围和失败边界。具体研究问题如下：")
    for item in [
        "RQ1：目标信念与欺骗信念是否需要分别建模，联合表示能否减少单一目标预测造成的混淆？",
        "RQ2：观察轨迹相对于诚实行为的反事实偏离能否提供可解释的欺骗证据？",
        "RQ3：如何根据不确定性、欺骗概率、观察成本与错误承诺风险选择 observe、track 或 commit？",
        "RQ4：trajectory-only 反欺骗在何种条件下不可辨识，增加模型容量或附加确定性特征是否能够突破该边界？",
        "RQ5：所提出方法能否在不同拓扑和不同信号机制的外部环境中保持优势？",
    ]:
        t.bullet(item)
    t.section("1.4 研究概述与技术路线")
    for text in [
        "本文把完整研究对象表示为“可见行为历史→目标与欺骗状态估计→信息可靠性评估→观察、跟踪或承诺→目标拦截”的闭环。7×7 PettingZoo ParallelEnv 将环境动力学、攻击者、信念估计器、控制器和评估器分离，真实目标只用于监督与离线评价，不进入防守策略的可见输入。攻击侧覆盖 honest、random、scripted、delayed、detour、multi-switch、stochastic 和 adaptive 等行为族，用于刻画固定诱导、长时诱导、组合策略和响应式欺骗。",
        "RIDe 的推理层将目标意图与欺骗状态作为相关但不同的隐变量：共享 GRU 提取时序表示，目标头输出 P(G=A|H_t) 与 P(G=B|H_t)，欺骗头输出 P(Z=1|H_t)。诚实轨迹原型和距离核产生 honest/deceptive likelihood ratio，校准器把原始概率转换为与经验正确率一致的置信度。决策层不直接复用分类标签，而是根据目标后验、信念熵、反事实证据、观察成本、错误承诺风险和剩余决策窗口选择 observe、track、commit_A 或 commit_B。",
        "实验验证围绕五个问题展开：策略性欺骗是否真实破坏轨迹推断；仅依赖可见历史时存在怎样的可辨识性上限；检测性能能否转化为拦截收益；主动感知与自适应承诺能否改善成本—成功权衡；方法在拓扑和观测语义变化后是否仍然有效。Bayesian、有限时域 POMDP、POMCP、Active Inference 与 MAPPO 构成传统推断、规划和学习基线；统一 DefenderEvaluationEnv 接口将 GridWorld、Forked Routes、PettingZoo signal deception 与 CyberDeceptionEnv 纳入同一指标体系。",
    ]:
        t.p(text)
    t.section("1.5 主要贡献")
    for item in [
        "提出联合目标信念、欺骗信念、反事实证据与自适应承诺的两层主动反欺骗框架。",
        "从检测—决策解耦角度系统展示高 AUROC 与低拦截率可以同时出现，并给出相应控制设计。",
        "形式化可观察别名命题，并通过平衡数据集和失败轨迹验证 trajectory-only 方法的经验上限。",
        "建立覆盖四环境、六方法、三攻击设置、五随机种子的 36,000 episode benchmark。",
        "提供从 reward hacking、belief shift、counterfactual misspecification 到 distribution shift 的完整失败归因。",
    ]:
        t.bullet(item)
    t.section("1.6 论文结构")
    t.p("第2章综述部分可观察决策、意图识别、策略性欺骗、主动感知与不确定性校准；第3章给出问题形式化、环境、攻击模型和评价指标；第4章阐述 RIDe 的双层信念、反事实推理与自适应承诺机制；第5章说明实验问题、数据、比较方法和统一协议；第6章分析欺骗学习中的奖励投机与训练失效；第7章验证可辨识性、检测—决策解耦和主动防守效果；第8章报告鲁棒性、泛化差距和失败归因；第9章给出多环境完整基准；第10章总结研究结论、理论意义、适用边界与未来方向。附录提供算法参数、分攻击族结果和复现信息。")


def add_chapter2(t: Thesis):
    t.chapter("第2章  相关研究与理论基础")
    sections = [
        ("2.1 部分可观察决策与信念状态", [
            "POMDP 使用隐藏状态、动作、转移、观察、奖励和折扣因子描述不完全信息下的序贯决策。历史可以压缩为关于隐藏状态的后验 belief；在模型正确时，belief 是决策的充分统计量。本文的隐藏变量不仅包含目标 G，还包含行为类型 Z，因此单一目标 posterior 不能完整描述欺骗风险。",
            "经典值迭代能够在离散 belief 空间上比较承诺和继续感知的期望价值，但复杂度随时域和观察分支指数增长。POMCP 通过粒子和蒙特卡洛树搜索近似在线规划，适合将 binary goal belief 与感知动作结合。本文同时实现精确有限时域 reduced POMDP 和轻量 POMCP，以区分模型失配与求解误差。",
        ]),
        ("2.2 意图识别与目标推断", [
            "目标推断根据位置、动作和上下文估计行动者要到达的目标。最短路径、Boltzmann rationality 和逆向规划是常用假设。它们在诚实导航中具有明确解释，但攻击者可以故意采取对错误目标更合理的动作，使似然模型产生高置信度错误后验。",
            "循环神经网络能够从可变长度轨迹中学习时序表示。本文采用 GRU，是因为数据规模有限且需要保持训练与推理成本可控。网络的目标不是替代环境，而是消费 Defender-visible history；这种模块化使规则、校准和学习估计器能够在同一环境中对比。",
        ]),
        ("2.3 策略性欺骗与对抗计划识别", [
            "策略性欺骗的关键在于攻击者选择行为以改变观察者的 belief。与静态对抗样本不同，轨迹欺骗具有时间结构：诱导动作会增加任务路径长度，反转时机决定防守者是否已承诺，自适应攻击还会根据已完成的观察和承诺动作改变后续轨迹。",
            "本文不把规则型 adaptive attacker 描述为博弈均衡。它只使用可公开的历史行为并根据固定规则调整诱导时间和目标，是对 response-conditioned behavior 的压力测试。该边界对于论文 claim 十分重要。",
        ]),
        ("2.4 主动感知与信息价值", [
            "主动感知不是被动接收观测，而是选择不同成本和精度的感知动作。observe 提供低成本、较窄的信息；track 提供更高成本、较强的线索。理性控制器应比较立即承诺的期望效用和额外观测带来的 posterior 改善。",
            "固定阈值可以阻止高熵下的承诺，但无法处理观察成本随时间累积、攻击者行为变化和即将错过拦截窗口等因素。本文因此从 uncertainty gate 进一步发展到风险感知和自适应观察策略。",
        ]),
        ("2.5 不确定性校准", [
            "神经网络 softmax 值不必等于真实正确概率。校准模型通过映射原始分数与经验正确率，使 confidence 更适合承诺决策。本文报告 Brier score、ECE 和 NLL，并将概率质量与拦截率分别分析。低校准误差并不自动意味着高任务成功，因为策略仍可能不 commit 或在错误时间 commit。",
        ]),
        ("2.6 强化学习与多智能体 PPO", [
            "PPO 使用 clipped surrogate objective 限制新旧策略比率，以提升策略梯度训练稳定性。MAPPO 通常采用共享或分散 actor 与集中式 critic。本文实现的 MAPPO actor 只接收统一的 observation 与 belief，critic 额外接收环境 one-hot 上下文。",
            "强化学习容易利用奖励定义中的捷径。若欺骗奖励可以在不完成目标时累计，攻击者可能停留或制造局部误导；若观察有成本而立即承诺的期望回报为零，防守者可能收敛到 A/B 固定猜测。本文将这些行为作为决策结构诊断，而不是简单归因于训练步数。",
        ]),
        ("2.7 研究定位", [
            "本文的创新不在于提出更大的轨迹分类器，而在于将欺骗感知明确建模为一个经过校准的序贯决策问题。目标与欺骗分别预测，诚实行为作为反事实参照，控制器显式选择信息获取和不可逆承诺。",
            "相较只报告分类准确率的工作，本文把 detection、commitment、interception、cost、latency、regret 和 calibration 放在同一评估中；相较只在单一网格验证的方法，本文进一步提供信号欺骗和网络防御环境，但也如实报告跨环境优势不一致。",
        ]),
    ]
    for title, paras in sections:
        t.section(title)
        for p in paras:
            t.p(p)


def add_chapter3(t: Thesis):
    t.chapter("第3章  问题定义与统一实验环境")
    t.section("3.1 双智能体部分可观察模型")
    for text in [
        "系统包含 Deceiver 和 Defender。真实目标 G∈{A,B} 在 episode 开始时确定，Deceiver 可以访问 G 及自身状态，Defender 只能访问局部地图、候选目标位置、可见轨迹和感知历史。Defender 的决策历史记为 H_t=(O_0:t,U_0:t−1)。",
        "Defender 动作集合统一为 U={observe, track, commit_A, commit_B}。observe 与 track 产生不同覆盖范围或不同可靠度的线索并承担成本；commit_A 与 commit_B 表示不可逆部署。导航环境中，承诺后防守者持续守卫相应目标，直至攻击者到达目标或时域结束；信号和网络环境在承诺时直接终止。",
        "攻击类型 Z 表示诚实或欺骗，但欺骗不是环境直接提供给策略的标签。训练阶段可以使用 Z 监督 deception head，评估和部署阶段只能根据 H_t 推断。",
    ]:
        t.p(text)
    t.equation("H_t=(O_{0:t},U_{0:t-1}),  G∈{A,B},  Z∈{0,1},  U_t∈{observe, track, commit_A, commit_B}")
    t.section("3.2 7×7 GridWorld")
    t.p("GridWorld 是研究链路的基础环境，采用 PettingZoo ParallelEnv。Deceiver 从固定起点出发，候选目标 A/B 对称布置。环境提供上下左右和停留五种移动动作；障碍、边界和最大 episode 长度共同定义可达轨迹。Defender 的局部传感器仅在攻击者进入覆盖区域时记录位置，未观察时刻不会泄漏真实位置。")
    t.p("信息边界测试检查 Defender observation 中不存在 true_goal，轨迹 mask 与可见时刻一致，环境不执行隐式 belief update。独立 estimator 接收可见历史并输出 {A:p_A,B:p_B}。这一设计是后续替换 progress estimator、GRU、双头模型和校准器的基础。")
    t.section("3.3 Forked Routes")
    t.p("Forked Routes 是具有分叉和障碍约束的第二导航环境。它与 GridWorld 共享有限离散运动模型、候选目标和 timed visible trajectory 接口，但路径拓扑不同。环境对称性检查确认两个目标的最短路径长度相同，因此 A/B 性能差异不能简单归因于几何距离。")
    t.section("3.4 PettingZoo deception environment")
    t.p("外部 PettingZoo 环境将空间轨迹抽象为信号序列。Deceiver 输出 claim A、claim B 或 mask，Defender 选择 observe、track 或 commit。track 触发具有固定准确率的审计线索。该环境用于检验方法是否依赖二维运动几何。")
    t.section("3.5 CyberDeceptionEnv")
    t.p("CyberDeceptionEnv 使用两个资产作为 A/B 目标，将位置观测替换为告警和 forensic cue。攻击设置包括 credential decoy、lateral switch、low-and-slow、observation-adaptive 与 stochastic 变体。Defender 的动作语义保持不变，从而可以使用同一 evaluator。")
    t.section("3.6 攻击族")
    rows = [
        ("Honest", "沿真实目标最短路径移动", "控制条件"),
        ("Scripted-k", "前 k 步朝错误目标，随后转向真实目标", "固定时序欺骗"),
        ("Delayed", "先表现诚实，再进入诱导阶段", "延迟反转"),
        ("Detour", "通过侧向或绕行动作形成错误几何证据", "路径偏离"),
        ("Multi-switch", "在真假目标之间多次切换", "非单次反转"),
        ("Stochastic", "按固定概率选择真假目标或随机动作", "未见随机族"),
        ("Adaptive", "根据 Defender 已完成的感知和承诺历史调整诱导", "行为自适应"),
    ]
    t.table("攻击族定义", ["攻击族", "行为机制", "用途"], rows, [3, 8, 4], 9)
    t.section("3.7 优化目标与指标")
    t.equation("J(π)=E_π[R_int−C_obs−λ_false·1{C≠G}]")
    t.p("拦截成功表示 Defender 最终守卫真实目标。false commitment 表示首次不可逆承诺与 G 不一致。observation cost 为 observe 与 track 成本之和。本文的 regret 以知道真实目标并立即正确守卫的特权 oracle 为参照：regret=1−(interception−cost)。它是统一评估量，不应解释为同信息条件下的 Bayes regret。")
    t.p("检测指标包括 deception accuracy、F1 与 AUROC；目标推断指标包括 goal accuracy、belief error 与 commitment accuracy；校准指标包括 Brier score、ECE 和 NLL。所有正式结果均注明 episode 数、随机种子和攻击协议。")


def add_chapter4(t: Thesis):
    t.chapter("第4章  RIDe 主动反欺骗框架")
    t.section("4.1 总体结构")
    t.p("RIDe 将防守分为推断层和决策层。推断层从 H_t 得到目标 belief、欺骗 probability、entropy 和 counterfactual likelihood ratio；决策层不修改推断模型，而根据这些量与成本选择感知或承诺。解耦使检测模型可以独立校准，也使 controller 的失败能够被单独诊断。")
    t.figure(ROOT / "outputs/stage15_paper/figures/main/detection_decision_gap.png", "检测质量与拦截决策之间的差异", 5.8)
    t.section("4.2 Deception-aware 双头信念模型")
    t.p("输入序列由归一化位置、时间、可见 mask 和可选 belief history 构成。GRU 逐步更新隐藏状态 h_t。共享表示分别进入 goal head 和 deception head。目标 head 使用二分类交叉熵，欺骗 head 使用 honest/deceptive 标签。联合目标为两项损失的加权和。")
    t.equation("h_t=GRU(x_t,h_{t-1})")
    t.equation("b_t=softmax(W_g h_t+c_g),    q_t=σ(W_z h_t+c_z)")
    t.equation("L_model=CE(b_t,G)+λ_z·BCE(q_t,Z)")
    t.p("共享编码器使目标方向和轨迹异常可以利用共同的时序特征，双 head 又避免把‘朝 A’和‘具有欺骗性’压缩为一个标签。训练/验证按 trajectory family 划分，避免同一完整轨迹的前缀同时进入不同 split。")
    t.algorithm("双头 GRU 的训练", [
        "输入：按 family 划分的可见轨迹数据 D，目标标签 G，欺骗标签 Z",
        "for epoch = 1,...,E do",
        "    对每个 padded sequence 计算 GRU 隐状态 h_t",
        "    计算 goal logits 与 deception logits",
        "    loss ← CE(goal_logits,G)+λ_z·BCE(deception_logits,Z)",
        "    反向传播并更新共享 encoder 与两个 head",
        "end for",
        "输出：冻结的双头 belief model",
    ])
    t.section("4.3 信念校准")
    t.p("独立 calibrator 作用于原始 goal probability。校准器不改变目标排序，而调整 confidence 与经验正确率的关系。controller 同时接收 P(G)、max confidence 和 entropy，避免仅凭 argmax 立即承诺。")
    t.equation("H(b_t)=−Σ_g b_t(g)log₂b_t(g)")
    t.section("4.4 反事实欺骗推理")
    t.p("反事实模块回答：如果行动者诚实地追求某个目标，当前轨迹应有多合理？实现首先从 honest 数据提取目标条件轨迹原型，再计算当前 prefix 与 honest/deceptive prototypes 的距离核兼容度。")
    t.equation("L(H_t)=P(H_t|Z=1)/P(H_t|Z=0)")
    t.equation("s_cf(H_t)=tanh(log L(H_t))")
    t.p("L>1 支持欺骗假设，L<1 支持诚实假设。该分数是可解释 baseline，不是真实生成密度。冻结模型的失败归因表明，scripted-8/10 在决策时可能与 honest reference 具有相同分数分布，说明原型假设会在长诱导和新行为族上失效。")
    t.section("4.5 自适应承诺控制")
    t.p("控制状态由 [P(A),P(B),entropy,counterfactual score,cumulative observation cost,commitment status] 构成。规则版本比较正确拦截收益、错误承诺风险和预期感知价值；学习版本使用一个两层 MLP，通过风险感知 teacher imitation 加 regret loss 学习四动作映射。")
    t.equation("U_commit(g)=b_t(g)R_int−(1−b_t(g))λ_false")
    t.equation("U_sense(u)=−c_u+E_y[max_g U_commit(g|τ(b_t,u,y))]")
    t.p("当 sensing utility 高于立即承诺时选择 observe 或 track；当目标不确定性低且承诺效用占优时选择 commit_A/B。承诺状态为 absorbing，后续策略输出不能更改守卫目标。")
    t.algorithm("RIDe 在线决策", [
        "输入：Defender-visible history H_t，累计成本 c_t，承诺状态 C_t",
        "若 C_t≠⊥：重复已承诺动作并返回",
        "使用双头 GRU 得到 b_t=P(G|H_t) 与 q_t=P(Z=1|H_t)",
        "校准 b_t，计算 entropy H(b_t)",
        "由 honest/deceptive prototype 计算 L(H_t) 与 s_cf",
        "构造决策特征 [b_t,H(b_t),s_cf,c_t,C_t]",
        "计算四动作效用或由冻结 MLP 输出动作概率",
        "施加 action mask；执行 observe、track、commit_A 或 commit_B",
        "更新历史、成本和不可逆承诺状态",
    ])
    t.section("4.6 可观察别名命题")
    t.p("命题1：若在事件 E 上，诚实与欺骗类型对 Defender-visible history 诱导相同条件分布，即 P(H|Z=0,E)=P(H|Z=1,E)，则任何只使用 H 的检测器在 E 上的条件 Bayes error 不低于 min{P(Z=0|E),P(Z=1|E)}。")
    t.p("证明：条件似然相等意味着所有 H 上的 likelihood ratio 为 1。由 Bayes 公式，观察 H 后的 posterior odds 与 prior odds 相同。最小错误分类器只能选择 prior majority class，并承担 minority prior mass。任何同一 σ(H) 上可测的模型都不能降低该误差。证毕。")
    t.p("该命题说明，当 timing 和 belief history 都是相同可见轨迹的确定函数时，附加这些特征不会增加信息。解决别名必须引入新的传感器、主动干预、环境上下文或更强行为假设。")


def add_chapter5(t: Thesis):
    t.chapter("第5章  实验设计与实现")
    t.section("5.1 实验问题与证据组织")
    questions = [
        ("欺骗有效性", "奖励、探索与梯度诊断", "判断高欺骗收益是否伴随真实任务完成，并识别停留、碰撞和目标不对称"),
        ("信息可辨识性", "轨迹、时序和信念历史对照", "测量相同可见历史对应不同标签时的经验上限"),
        ("检测—决策关系", "被动分类、主动规则与信念规划", "检验高 F1/AUROC 是否能转化为正确且及时的拦截"),
        ("主动决策价值", "校准、反事实证据与自适应观察", "比较拦截成功、错误承诺、观察成本和 regret"),
        ("泛化与鲁棒性", "未见攻击、第二拓扑和外部环境", "定位行为分布偏移、反事实失配与观测语义变化"),
        ("完整方法比较", "四环境、六方法、三攻击设置", "在统一接口、固定预算和五随机种子下验证结论"),
    ]
    t.table("研究问题与实验证据", ["研究问题", "实验设计", "证据目标"], questions, [3.0, 5.5, 6.8], 9)
    t.section("5.2 数据集构造")
    for text in [
        "基础数据记录 Defender-visible trajectory、trajectory mask、action timing、true goal 和 deception family。训练集包含 Honest 与 Scripted steps 1–4，测试覆盖 steps 5、delayed、detour 和 multi-switch。可辨识性实验使用标签平衡的数据集，使 honest/deceptive 数量相等，并按 trajectory family 而不是随机轨迹划分训练与验证，从而避免同族模板同时出现在训练集和测试集。",
        "输入模式 A 只使用 trajectory；模式 B 增加 action timing；模式 C 增加 belief history。三个模式使用相同 GRU 结构，以判断新增特征是否提供独立信息。结果三者 accuracy、F1 和 AUROC 完全相同，表明附加量在构造数据中是轨迹的确定函数。",
        "主动防守数据把每个前缀表示为目标 belief、欺骗概率、信念熵、历史嵌入与累计观察成本。承诺标签来自规则控制器或特权 teacher，但评估时不暴露 true goal。未见攻击集包含长时脚本诱导、delay+detour、detour+switch、随机欺骗和更强的响应式攻击，用于检验模型是否只记忆训练族的反转时刻。",
    ]:
        t.p(text)
    t.section("5.3 比较方法")
    methods = [
        ("Rule Active", "固定 evidence、稳定性与熵规则；主动 observe/track"),
        ("Bayesian", "利用 analytic belief 或有限状态 honest-motion likelihood 做 posterior stopping"),
        ("POMDP/POMCP", "在二元目标 belief 上比较感知和承诺的有限时域价值"),
        ("Active Inference", "最小化期望自由能近似，结合预期熵和动作成本"),
        ("MAPPO", "共享 actor、环境上下文 centralized critic，PPO clipped update"),
        ("RIDe", "冻结双层信念、学习型决策 checkpoint 与统一外部适配"),
    ]
    t.table("正式 benchmark 方法", ["方法", "实现要点"], methods, [3.2, 11.8], 9)
    t.section("5.4 多环境完整评估协议")
    t.p("正式 benchmark 包含 4 个环境、6 个方法和 3 个攻击设置。使用 seed 0–4；每个 seed、方法、环境和攻击设置运行 100 episodes，A/B 各 50。因此每个方法—环境—攻击条件聚合 500 episodes，总评估量为 5×6×4×3×100=36,000 episodes。")
    t.p("MAPPO 对每个 seed 独立训练约 100,000 timesteps，总预算约 500,000 timesteps。训练仅使用 scripted 和 adaptive 条件，unseen family 不进入训练。每次 rollout 约 2,048 transitions，PPO 使用 4 epochs、minibatch 256、学习率 3×10⁻⁴、γ=0.99、clip range 0.2 和 entropy coefficient 0.01。")
    t.p("主要决策指标包括 interception success、false commitment、observation cost、decision regret 和 decision latency；推断指标包括 goal accuracy、deception F1、AUROC、Brier score、ECE 与 NLL。所有聚合结果先在每个随机种子内部计算，再报告五个种子的 mean±std，避免把大量相关回合误当作独立样本。")
    t.section("5.5 可复现性与保护机制")
    t.p("每类实验保存 episode-level CSV、summary、config、checkpoint hash 和 preserved_inputs.json。正式 runner 在执行前后计算 GridWorld、Forked Routes、学习型 controller、双头 belief model 和 calibrator 的 SHA-256；若任一受保护输入改变则终止。完整基准的 preserved_inputs 记录显示所有冻结对象前后一致。")
    t.p("测试覆盖环境 API、信息边界、动作 mask、随机种子确定性、训练 smoke、CSV schema 与聚合 episode 数。最终仓库回归测试为 146 passed。")


def add_chapter6(t: Thesis):
    t.chapter("第6章  欺骗学习的有效性与训练失效分析")
    t.section("6.1 信念输入与奖励课程消融")
    t.p("为判断攻击者是否真正利用防守者信念，而非只记忆环境动作，实验比较 PPO-no-belief、PPO-with-belief 和 PPO-with-belief+curriculum。前两组固定 reward=task+deception+cost，课程组在训练过程中使用 task only、task+deception 和 task+deception+cost 三种目标。三组共享 Defender Model A、随机种子、step_commit、PPO 超参数和评估回合数，使结果差异只对应 belief 输入和奖励课程。")
    t.figure(ROOT / "outputs/ppo_stage72/learning_curves.png", "信念输入与奖励课程的 PPO learning curves", 5.9)
    t.figure(ROOT / "outputs/ppo_stage72/belief_ablation.png", "belief observation 消融结果", 5.9)
    t.p("训练曲线表明 deception shaping 可以持续增长，而 task completion 没有同步恢复；belief-aware observation 也未自动带来更高 task success。这一差异说明，策略能够利用 belief 作为获取局部欺骗奖励的信号，却没有被迫学习“误导之后仍到达真实目标”的完整行为。")
    t.section("6.2 任务奖励与欺骗奖励消融")
    t.p("R1 提高 task success reward 并降低 deception reward；R2 保持原比例；R3 只在接近目标或成功后给予 delayed deception bonus。所有条件使用 50k timesteps、seed 0、A/B 各 25 的 50-episode 评估。")
    t.figure(ROOT / "outputs/ppo_reward_ablation/learning_curves.png", "三种奖励定义下的训练曲线", 5.9)
    t.figure(ROOT / "outputs/ppo_reward_ablation/learned_trajectory_examples.png", "PPO 学习轨迹与异常行为示例", 5.9)
    t.p("三种配置整体 task success 均为 50%，但目标条件结果表现为 A 0%、B 100%。这一结构在 reward 比例变化后保持不变，说明根因不是单一 shaping 系数。失败轨迹包含 stay、边界重复动作和成功误导但不完成任务。")
    t.section("6.3 探索强度与评估随机性")
    t.p("探索诊断将 entropy coefficient 提高五倍，并对同一 checkpoint 分别使用 deterministic=True 与 deterministic=False 评估。随机动作能够偶尔离开固定点，但并未稳定恢复 left action 或 A 目标成功。确定性评估会把策略分布中的小概率偏差放大为固定动作，却不是 A/B 不对称的唯一来源；较高熵同样没有产生覆盖目标 A 的有效轨迹。")
    t.section("6.4 rollout coverage 与 advantage")
    t.p("梯度诊断统计 A/B rollout 数量、成功 episode、到达 A 附近的轨迹和 left action 次数，并比较 left/right/up/down/stay advantage。A 相关成功轨迹稀少，left action 的正 advantage 信号不足；critic explained variance 和 actor loss 虽发生变化，却没有转化为状态覆盖改善。因而策略坍塌既包含探索覆盖不足，也包含有用动作缺少稳定正优势的问题。")
    t.section("6.5 结论：训练预算不是充分条件")
    t.p("在多环境基准中，MAPPO 的预算扩大至每个 seed 约 100k timesteps。五条 deterministic validation curve 最终都稳定为 0.5，策略在所有环境选择零观察成本的即时承诺。该结果说明，只增加样本量而不改变信息价值建模与承诺结构，无法保证主动防守行为涌现；训练指标收敛只能证明优化过程稳定，不能证明策略学会了利用信息。")
    t.figure(ROOT / "outputs/stage17_3_full_benchmark/figures/mappo_training_curves.png", "五 seed MAPPO 训练与收敛曲线", 6.0)


def add_chapter7(t: Thesis):
    t.chapter("第7章  欺骗推理与主动决策实验")
    t.section("7.1 双任务信念估计")
    t.p("欺骗检测数据集以 Defender-visible trajectory 为输入，同时提供 A/B 目标标签与 honest/deceptive 行为标签。训练族为 Honest 与 Scripted steps 1–4，测试族为 steps 5、delayed、detour 和 multi-switch。共享 GRU 通过同一时序表示预测两个隐变量，使目标不确定性与欺骗不确定性能够分别度量。该设置检验的不是单纯多任务准确率，而是双头输出能否为后续控制器提供语义清晰的目标后验和欺骗概率。")
    t.section("7.2 轨迹信息的可辨识性边界")
    t.p("平衡数据集包含 honest、steps 1–5、delayed、detour 和 multi-switch。A/B/C 三种输入模式得到相同的 deception accuracy、F1=0.8966 和 AUROC=0.9362。有限数据中存在 32 个相同 observation 对应不同 deception label 的不可约错误，经验 accuracy ceiling 为 93.75%。")
    t.p("action timing 和 belief history 没有带来提升，是因为二者在该数据构造中都是同一可见轨迹的确定函数，并未增加新的随机变量。对于完全相同的可见前缀，诚实与欺骗解释具有相同输入，任何确定性分类器都必须给出相同输出。该结果与第4章的可观察别名命题一致，说明扩大 GRU 隐层或延长训练不能消除由信息边界造成的误差。")
    t.figure(ROOT / "outputs/stage15_paper/figures/appendix/identifiability_modes.png", "三种输入模式下的可辨识性结果", 5.8)
    t.section("7.3 被动检测与主动防守")
    for text in [
        "被动基线只根据轨迹输出 deception probability，不能改变后续信息量，也不能直接决定拦截位置。主动 Defender 将动作扩展为 observe、track、commit_A 和 commit_B，使系统能够通过付出成本换取新观测。规则控制和简单 Q-learning 的对照表明，主动获取信息可以减少部分错误承诺，但动作效用必须同时考虑剩余决策窗口，否则持续观察同样会错失目标。",
        "以 goal belief、deception probability、entropy 和 history embedding 为状态的学习策略出现两种互补的奖励捷径：观察成本会诱导立即 commit，而较强的 entropy-weighted commitment penalty 会诱导永不 commit。硬不确定性门控能够禁止高熵承诺，却仍依赖人工阈值。上述对照说明，立即行动与无限等待不是简单的数值异常，而是风险、成本和信息价值没有被共同建模时的结构性失效。",
    ]:
        t.p(text)
    t.section("7.4 欺骗感知推理与决策解耦")
    t.p("RIDe 将 deception-aware belief、counterfactual reasoning 和 adaptive commitment 组织为推理层与决策层。双头模型在 seen/unseen 数据上的 detection F1 达 93.33%，但直接把推理输出映射为承诺时，task success 只有 37.5%。这组结果说明检测器可以正确判断“行为可能具有欺骗性”，却仍无法独立回答“真实目标是什么”和“现在是否应该承诺”。")
    t.figure(ROOT / "outputs/stage14_paper_suite/figures/decision_ablation.png", "推断—决策结构消融", 5.9)
    t.section("7.5 概率校准与风险感知承诺")
    t.p("在冻结 dual-head GRU 和 counterfactual prototypes 的条件下，校准器调整目标概率与经验正确率之间的对应关系，风险控制器则比较 expected interception reward、observation cost 与 false commitment risk。相同 128-episode 协议下，Rule Active 与解耦控制器的 interception 均为 93.75%，而直接耦合的 Two-level controller 只有 37.5%。这组结构消融支持“检测输出需要经过校准和决策效用转换”的主张，但校准、解耦与风险计算在该实验中同时改变，因此不能把全部增益归于其中某一个组件。")
    t.section("7.6 自适应观察的成本收益")
    t.p("在 delayed、detour 和 multi-switch 攻击上，Rule Active、固定风险控制器和 Adaptive Defender 的拦截率均为 100%。Adaptive Defender 的 observation cost 为 0.1083，低于 Rule Active 的 0.1667 和固定风险控制器的 0.1983。由于成功率相同，成本下降可以直接归因于观察动作随不确定性和欺骗似然动态变化；该证据只覆盖受控攻击协议，不意味着相同策略可无损迁移到任意观测语义。")
    t.section("7.7 响应式攻击压力测试")
    t.p("AdaptiveDeceiver 包含 commitment-aware、target-aware 和 observation-aware 三类规则。它根据历史 commit time 调整诱导窗口，根据长期目标偏好选择假目标，并在 Defender observe/track 后改变 fake trajectory。更强版本加入更宽承诺窗口、时序 jitter 和多次切换，用于检验固定观察模式是否可被攻击者预测。攻击者不使用 reward optimizer，因此这些结果证明的是 response-conditioned behavior 下的鲁棒性，不是博弈均衡或最优攻击。")
    t.section("7.8 经典信念规划与检测—决策鸿沟")
    t.p("在统一协议中，formal Bayesian 在 Honest 上实现 100% 拦截，在 Scripted-3 与 Adaptive attacker 上均为 0%；POMDP 在 Honest 为 100%，Scripted-3 为 0%，Adaptive 为 5.88%。两者在欺骗条件下 AUROC 可达 1.0，却因 honest-motion likelihood 被 feint 诱导而过早错误承诺。这里的失败不来自检测器无法识别异常，而来自目标生成模型仍把诱导动作解释为错误目标证据。")
    t.figure(ROOT / "outputs/stage15_paper/figures/main/detection_decision_gap.png", "传统信念基线中的检测—决策鸿沟", 5.8)
    t.section("7.9 学习型决策控制器")
    t.p("学习型控制器用风险感知规则作为 teacher，训练一个 1,412 参数的两层 MLP。输入为六维决策状态，损失由 cross-entropy imitation 与 expected teacher-utility regret 组成。在 128 个 adaptive attacker 场景中，teacher 与学习控制器均达到 100% 拦截和 0% false commitment；学习控制器将 observation cost 从 0.3499 降至 0.1966。该实验说明小模型能够压缩并平滑 teacher 的决策逻辑，但由于 belief encoder 保持冻结，结果不构成独立端到端泛化证据。")


def add_chapter8(t: Thesis):
    t.chapter("第8章  鲁棒性评估与失败归因")
    t.section("8.1 共同协议下的方法对比")
    t.p("内部对比将主要方法、消融条件、失败案例和理论分析映射到同一份可复现输出。探索性主表采用 seed 0，每个 method—attack cell 34 episodes，A/B 各 17。由于样本规模较小，该部分只用于描述控制器行为和筛查潜在失效，不承担最终跨环境统计结论。")
    t.figure(ROOT / "outputs/stage14_paper_suite/figures/main_interception.png", "共同协议下的拦截率", 5.9)
    t.figure(ROOT / "outputs/stage14_paper_suite/figures/cost_success_tradeoff.png", "观察成本与拦截成功权衡", 5.9)
    t.section("8.2 多随机种子鲁棒性与拓扑迁移")
    t.p("在算法和 checkpoint 保持冻结的条件下，五个随机种子覆盖 stronger adaptive attacker v2、scripted 8/10、delay+detour、detour+switch 和 stochastic deception，并在 Forked Routes 上检验拓扑迁移。GridWorld 中学习型控制器对 adaptive v1/v2 均达到 100%，但对 scripted 8/10 为 0%，对 stochastic deception 为 44.0%±11.4%。结果说明，响应式攻击的表面复杂度不必然高于长时固定诱导；当长诱导保持与诚实路径相似的前缀时，冻结表示反而更容易发生系统性误判。")
    t.figure(ROOT / "outputs/stage15_robustness/figures/stage13_five_seed_robustness.png", "学习型控制器的五 seed 鲁棒性", 5.9)
    t.figure(ROOT / "outputs/stage15_robustness/figures/forked_routes_main.png", "Forked Routes 正式结果", 5.9)
    t.p("Forked Routes 显示明显 transfer gap。学习型控制器在 scripted-10 上为 0%，adaptive-v2 上仅 5%；冻结的风险感知规则在多个确定性 family 上接近 50%。相同动作接口只保证评估语义一致，不保证轨迹编码器已经学到跨拓扑不变的行为表示。")
    t.section("8.3 目标信念的泛化差距")
    t.p("冻结重放把 Scripted-3 定义为 seen，把 scripted 8/10、delay_detour、detour_switch 和 stochastic 定义为 unseen。决策时 belief error 从 seen 的 0.0125±0.0000 增至 unseen 的 0.6465±0.0215，差值为 +0.6340±0.0215；goal accuracy 从 1.000 降至 0.288。该差距说明最终拦截下降主要发生在承诺之前的目标信念环节，而不是只有控制器动作映射发生变化。")
    t.figure(ROOT / "outputs/stage16_failure_attribution/belief_generalization_gap.png", "seen 与 unseen 的 belief error 差异", 5.8)
    t.section("8.4 counterfactual score 分布")
    t.p("seen counterfactual reference interval 为 [−0.0913,0.6618]。Scripted-8/10 的分布与 honest reference 完全相同：mean −0.0907，fraction nonpositive=1.0。系统因此把长诱导误认为诚实行为。delay_detour、detour_switch 和 stochastic 的均值为正，但分别有 31.6%、22.2% 和 34.6% 状态不支持欺骗假设。")
    t.figure(ROOT / "outputs/stage16_failure_attribution/counterfactual_score_distribution.png", "不同攻击族的反事实分数分布", 5.8)
    t.section("8.5 决策边界")
    t.p("决策边界图固定累计 observation cost 为训练轨迹的 precommit median 0.09，只展示六维控制器在 belief confidence 与 counterfactual score 两个维度的切片。边界用于解释模型如何把高置信度与欺骗证据映射为 observe/track/commit，不能视为完整策略的充分描述。")
    t.figure(ROOT / "outputs/stage16_failure_attribution/decision_boundary.png", "学习型 controller 的二维决策边界切片", 5.8)
    t.section("8.6 failure taxonomy")
    t.p("712 个失败 deceptive episodes 全部属于 attacker distribution shift；433 个满足严格 observational aliasing，3 个满足 premature commitment，421 个在决策时 counterfactual likelihood ratio≤1。机制可以重叠。按 aliasing→premature→wrong counterfactual→distribution shift 的固定优先级，primary counts 为 aliasing 433、premature 1、wrong counterfactual 8、distribution shift 270。")
    t.figure(ROOT / "outputs/stage16_failure_attribution/failure_taxonomy.png", "失败类型及重叠机制", 5.8)
    t.p("该 taxonomy 是基于可观察规则的诊断，不是严格因果识别。存在完全相同 history 是信息不足的充分证据；不存在完全匹配并不能证明任务可辨识。")


def add_chapter9(t: Thesis):
    main = read_csv(ROOT / "outputs/stage17_3_full_benchmark/main_results.csv")
    t.chapter("第9章  多环境完整基准评估")
    t.section("9.1 统一接口")
    t.p("四个环境统一暴露 observation、belief、action、cost 和 terminal success。所有方法支持 observe、track、commit_A 和 commit_B。GridWorld 与 Forked Routes 通过导航 adapter 转换 Defender action；PettingZoo 和 Cyber 直接使用信号/告警 observation。")
    t.section("9.2 攻击设置与训练边界")
    t.p("Scripted 在导航环境采用 scripted-3，在 PettingZoo 采用固定前四步错误 claim，在 Cyber 采用 credential decoy。Adaptive 根据 Defender 已完成动作调整诱导。Unseen family 使用预先声明的 stochastic deception，并从 MAPPO 训练数据中排除。")
    t.section("9.3 主结果")
    order_methods = ["rule_active", "bayesian", "pomcp", "active_inference", "mappo", "ride"]
    order_env = ["gridworld", "forked_routes", "pettingzoo_deception", "cyber_deception"]
    lookup = {(r["method"], r["environment"]): r for r in main}
    rows = []
    for method in order_methods:
        rows.append([method] + [lookup[(method, env)]["interception_mean_std"] for env in order_env])
    t.table("跨攻击族拦截率（mean±std across seeds）", ["方法", "GridWorld", "Forked", "PettingZoo", "Cyber"], rows, [3.2, 2.6, 2.6, 2.6, 2.6], 8.5)
    t.figure(ROOT / "outputs/stage17_3_full_benchmark/figures/main_interception.png", "六方法在四环境中的拦截率", 6.2)
    t.p("RIDe 在 GridWorld 达到 0.823±0.023，高于 POMCP 的 0.487±0.026 和 Rule Active 的 0.124±0.019；在 Cyber 达到 0.980±0.008，高于 Rule 的 0.895±0.019。Forked Routes 上 RIDe 为 0.569±0.006，优势有限。PettingZoo 上 RIDe 只有 0.523±0.029，低于 Rule Active 的 0.792±0.017、Bayesian 的 0.745±0.025 和 Active Inference 的 0.740±0.028。")
    t.p("因此，正式结果支持环境相关的优势，而不支持普遍最优。RIDe 更适合其反事实/信念表示能够捕获行为结构的 GridWorld 和 Cyber；在信号机制不同的 PettingZoo 中，固定规则与解析 belief 更有效。")
    t.section("9.4 regret 与 observation cost")
    rows = []
    for method in order_methods:
        values = [lookup[(method, env)] for env in order_env]
        rows.append((method, *[value["regret_mean_std"] for value in values]))
    t.table("多环境 decision regret", ["方法", "GridWorld", "Forked", "PettingZoo", "Cyber"], rows, [3.2, 2.6, 2.6, 2.6, 2.6], 8.5)
    t.figure(ROOT / "outputs/stage17_3_full_benchmark/figures/main_regret.png", "六方法在四环境中的 decision regret", 6.2)
    t.p("RIDe 在提高部分环境拦截率的同时承担较高观察成本：GridWorld 0.241、Forked 0.263、PettingZoo 0.200、Cyber 0.273。MAPPO observation cost 为 0，因为其 deterministic policy 立即 commit；其 regret 固定 0.5。零成本不是效率优势，而是策略坍塌的表现。")
    t.section("9.5 calibration")
    t.figure(ROOT / "outputs/stage17_3_full_benchmark/figures/calibration_comparison.png", "六方法的期望校准误差比较", 5.9)
    t.p("RIDe 在 Cyber 和 PettingZoo 的 Brier score 分别为 0.019 和 0.024，说明其最终 belief 与真实目标高度一致；但 PettingZoo 拦截率只有 52.3%。该差异再次证明 belief calibration 与 action success 是互补指标：正确 belief 可能在承诺覆盖、时机或动作映射上未被充分利用。Forked Routes 的 RIDe Brier 为 0.417，反映明显跨拓扑失配。")
    t.section("9.6 分攻击族结果")
    t.figure(ROOT / "outputs/stage17_3_full_benchmark/figures/per_family_interception.png", "scripted、adaptive 与 unseen family 分组结果", 6.2)
    t.p("分 family 曲线用于区分整体平均背后的攻击差异。所有 family 每个 cell 均包含 500 episodes，因此不会因样本数不同而产生加权偏差。unseen 不用于 MAPPO 训练，RIDe 与传统方法也不根据完整基准结果重新调参。")
    t.section("9.7 结果解释")
    for text in [
        "方法优势明显依赖 observation semantics。GridWorld 的几何反转和 Cyber 的告警/forensic 证据可被 RIDe 的 belief 与反事实代理利用；PettingZoo 的离散 claim 更适合直接解析更新。该差异解释了 RIDe 为何能在部分环境领先，却在 PettingZoo 中落后于规则与解析方法。",
        "规划质量受到生成模型准确性的限制。POMCP 在四环境取得 48.7%–61.9%，比 honest-motion Bayesian 在欺骗严重的 GridWorld 更稳健，但仍受固定 cue accuracy 和 reduced belief model 约束。更复杂的规划器无法补偿系统性错误的行为似然。",
        "训练收敛也不等同于策略有效。五个 MAPPO seed 的最后五个 validation window 均为 0.5，absolute change 为 0，按 0.02 tolerance 被标记为收敛；它们收敛到的是错误的即时承诺解。这一现象与第6章的奖励投机分析相互印证。",
        "跨环境适配还需要区分 checkpoint transfer 与接口复用。RIDe external adapter 保持学习型控制器权重冻结，并用 belief confidence/reversal 构造固定 counterfactual proxy。结果验证决策 checkpoint 的外部适配，而不是原始 GridWorld GRU 在所有 observation domain 上的端到端零样本迁移。",
    ]:
        t.p(text)


def add_chapter10(t: Thesis):
    t.chapter("第10章  讨论、结论与展望")
    t.section("10.1 研究结论")
    for text in [
        "本文证明，策略性欺骗下的防守应被建模为经过校准的序贯决策问题，而非单纯 trajectory classification。目标 belief、欺骗 belief 和行动控制承担不同功能，必须分别评估。",
        "反事实诚实行为提供了可解释的欺骗证据，但其可靠性取决于 prototype 是否覆盖当前攻击族。自适应 commitment 能在受控 family 上降低 observation cost，同时长诱导、随机行为和新拓扑会破坏这一假设。",
        "完整 benchmark 显示 RIDe 在 GridWorld 和 Cyber 上取得明显优势，在 Forked Routes 上略有优势，在 PettingZoo 上落后于规则方法。因此本文的最终 claim 是有边界的：RIDe 能在若干 observation domain 中改善主动反欺骗决策，但不是跨环境通用最优方法。",
    ]:
        t.p(text)
    t.section("10.2 理论意义")
    t.p("可观察别名命题把分类失败与信息边界联系起来。当不同类型产生相同 history 时，任何同信息输入的深度模型都不能突破 Bayes 下界。该结论提示后续工作应优先设计新观测或主动干预，而不是只扩大网络。")
    t.section("10.3 工程意义")
    t.p("统一 DefenderEvaluationEnv 使导航、信号和网络防御环境共享动作和指标。环境、belief、controller 与 evaluator 的解耦减少了隐藏标签泄漏，并允许冻结 checkpoint 的重复验证。保存逐 episode 数据和 hash 为论文 claim 审计提供依据。")
    t.section("10.4 局限性")
    limitations = [
        "攻击者仍以规则为主，没有学习到博弈均衡，也不能代表最优战略对手。",
        "部分探索性实验为单 seed、小样本验证，其数值不应与多环境正式结果直接合并。",
        "GridWorld 和 Forked Routes 的轨迹可能高度重复，episode 数大于有效独立轨迹数。",
        "RIDe external adapter 使用固定 counterfactual proxy，尚未完成统一 encoder 的跨模态迁移。",
        "regret 参照知道真实目标的 oracle，不是相同信息条件下的 Bayes regret。",
        "本文的 MAPPO 共享 actor/centralized critic 是轻量 baseline，不代表所有 MAPPO 架构。",
        "长 scripted feint、observational aliasing 和 counterfactual misspecification 尚未解决。",
    ]
    for item in limitations:
        t.bullet(item)
    t.section("10.5 未来工作")
    for text in [
        "未来可引入主动询问、诱饵部署和传感器重定位，使 Defender 的动作不仅改变观测质量，也改变攻击者可选行为，从信息源上打破 aliasing。",
        "可使用层次 Bayesian model 或基于不确定性的 ensemble 对 honest/deceptive dynamics 进行在线更新，并将 counterfactual assumption 的可信度纳入 controller。",
        "在学习方面，应研究具有 commitment option、risk constraint 和 explicit information-gain objective 的 actor-critic，避免 observation cost 驱动的立即承诺。",
        "在验证方面，需要更多地图、真实网络日志和机器人轨迹，并采用预注册协议、置信区间和显著性检验。对于 adaptive attacker，应加入双边在线学习并分析 exploitability，而非只报告单向适应。",
    ]:
        t.p(text)
    t.section("10.6 总结")
    t.p("本文把策略性反欺骗表述为推断与决策相互作用的序贯问题，并提出结合目标—欺骗双层信念、反事实行为比较和自适应承诺的 RIDe 框架。实验同时报告成功结果和失效证据，通过 reward diagnosis、identifiability、decision boundary、failure taxonomy 与 external benchmark 说明方法在何处有效、为何失效以及哪些问题在当前信息条件下不可解决。除算法设计外，这种以可辨识性、决策效用和适用边界组织证据的分析框架也是本文的重要成果。")


def add_references(t: Thesis):
    t.chapter("参考文献")
    references = [
        "[1] Kaelbling L P, Littman M L, Cassandra A R. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 1998, 101(1–2): 99–134.",
        "[2] Silver D, Veness J. Monte-Carlo planning in large POMDPs. Advances in Neural Information Processing Systems, 2010.",
        "[3] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.",
        "[4] Terry J K, Black B, Grammel N, et al. PettingZoo: Gym for multi-agent reinforcement learning. Advances in Neural Information Processing Systems, 2021.",
        "[5] Cho K, van Merriënboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP, 2014.",
        "[6] Hochreiter S, Schmidhuber J. Long short-term memory. Neural Computation, 1997, 9(8): 1735–1780.",
        "[7] Guo C, Pleiss G, Sun Y, Weinberger K Q. On calibration of modern neural networks. ICML, 2017.",
        "[8] Gal Y, Ghahramani Z. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. ICML, 2016.",
        "[9] Settles B. Active Learning Literature Survey. University of Wisconsin–Madison, 2009.",
        "[10] Sutton R S, Barto A G. Reinforcement Learning: An Introduction. 2nd ed. MIT Press, 2018.",
        "[11] Puterman M L. Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley, 1994.",
        "[12] Harsanyi J C. Games with incomplete information played by Bayesian players. Management Science, 1967–1968.",
        "[13] Shannon C E. A mathematical theory of communication. Bell System Technical Journal, 1948.",
        "[14] Cover T M, Thomas J A. Elements of Information Theory. 2nd ed. Wiley, 2006.",
        "[15] Goodfellow I, Shlens J, Szegedy C. Explaining and harnessing adversarial examples. ICLR, 2015.",
        "[16] Ng A Y, Russell S. Algorithms for inverse reinforcement learning. ICML, 2000.",
        "[17] Ramírez M, Geffner H. Plan recognition as planning. IJCAI, 2009.",
        "[18] Baker C L, Saxe R, Tenenbaum J B. Action understanding as inverse planning. Cognition, 2009, 113(3): 329–349.",
        "[19] Friston K, FitzGerald T, Rigoli F, et al. Active inference: A process theory. Neural Computation, 2017, 29(1): 1–49.",
        "[20] Pearl J. Causality: Models, Reasoning, and Inference. 2nd ed. Cambridge University Press, 2009.",
        "[21] Murphy K P. Probabilistic Machine Learning: An Introduction. MIT Press, 2022.",
        "[22] Bishop C M. Pattern Recognition and Machine Learning. Springer, 2006.",
        "[23] Good I J. Rational decisions. Journal of the Royal Statistical Society B, 1952, 14(1): 107–114.",
        "[24] DeGroot M H. Optimal Statistical Decisions. Wiley, 1970.",
        "[25] Lowe R, Wu Y, Tamar A, et al. Multi-agent actor-critic for mixed cooperative-competitive environments. NeurIPS, 2017.",
        "[26] Yu C, Velu A, Vinitsky E, et al. The surprising effectiveness of PPO in cooperative multi-agent games. NeurIPS, 2022.",
        "[27] Foerster J, Farquhar G, Afouras T, et al. Counterfactual multi-agent policy gradients. AAAI, 2018.",
        "[28] Oliehoek F A, Amato C. A Concise Introduction to Decentralized POMDPs. Springer, 2016.",
        "[29] Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. ICML, 2005.",
        "[30] Dawid A P. The well-calibrated Bayesian. Journal of the American Statistical Association, 1982, 77(379): 605–610.",
        "[31] Vrij A. Detecting Lies and Deceit: Pitfalls and Opportunities. 2nd ed. Wiley, 2008.",
        "[32] Albrecht S V, Stone P. Autonomous agents modelling other agents: A comprehensive survey and open problems. Artificial Intelligence, 2018, 258: 66–95.",
        "[33] Doshi-Velez F, Kim B. Towards a rigorous science of interpretable machine learning. arXiv:1702.08608, 2017.",
        "[34] Henderson P, Islam R, Bachman P, et al. Deep reinforcement learning that matters. AAAI, 2018.",
        "[35] Agarwal R, Schwarzer M, Castro P S, et al. Deep reinforcement learning at the edge of the statistical precipice. NeurIPS, 2021.",
    ]
    for reference in references:
        p = t.doc.add_paragraph(reference)
        p.paragraph_format.first_line_indent = Cm(-0.75)
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(4)


def add_appendices(t: Thesis):
    t.chapter("附录A  关键算法与参数")
    t.section("A.1 Progress-based belief baseline")
    t.p("规则 estimator 比较最新可见位置相对两个目标的距离和最近一步 progress，并对两个 score 做 softmax。它不访问 true goal，主要用于环境与 evaluator 的早期验证。")
    t.algorithm("Progress-based belief update", [
        "对每个候选目标 g 计算 distance d_g(x_t)",
        "若存在连续可见点，计算 progress Δ_g=d_g(x_{t-1})−d_g(x_t)",
        "score_g=−0.55·d_g(x_t)+1.10·Δ_g",
        "b_t=softmax(score_A,score_B)",
    ])
    t.section("A.2 Bayesian posterior stopping")
    t.algorithm("Bayesian Defender", [
        "初始化 b_0(A)=b_0(B)=0.5",
        "根据 honest-motion transition likelihood 更新 log belief",
        "若 t≥earliest 且 max_g b_t(g)≥confidence threshold：commit argmax b_t",
        "否则若 t≥deadline：commit argmax b_t",
        "否则继续 observe/track",
    ])
    t.section("A.3 POMCP online planning")
    t.algorithm("Reduced-belief POMCP", [
        "从当前 P(G=B) 采样 hidden-goal particles",
        "对 observe、track、commit_A、commit_B 建立在线搜索树",
        "感知分支按固定 cue accuracy 采样并做 Bayes update",
        "使用 UCB 在节点选择动作，回传 reward−cost",
        "根节点选择 visit/value 最大且 action mask 允许的动作",
    ])
    t.section("A.4 MAPPO 训练参数")
    params = [
        ("训练 seed", "0,1,2,3,4"),
        ("timesteps/seed", "100,000（实际完整 episode 略有超出）"),
        ("rollout steps", "2,048"),
        ("PPO epochs", "4"),
        ("minibatch", "256"),
        ("learning rate", "3×10⁻⁴"),
        ("gamma", "0.99"),
        ("clip range", "0.2"),
        ("entropy coefficient", "0.01"),
        ("actor input", "8-d observation + 2-d belief"),
        ("critic context", "4-d environment one-hot"),
        ("训练攻击", "scripted + adaptive"),
        ("留出攻击", "unseen_family"),
    ]
    t.table("多环境基准中的 MAPPO 配置", ["参数", "取值"], params, [5, 10], 9.5)

    t.chapter("附录B  分攻击族完整结果")
    data = read_csv(ROOT / "outputs/stage17_3_full_benchmark/per_family_results.csv")
    rows = [
        (
            r["method"],
            r["environment"],
            r["attack_family"],
            r["interception_mean_std"],
            r["regret_mean_std"],
            r["observation_cost_mean_std"],
            r["calibration_mean_std"],
        )
        for r in data
    ]
    t.table("六方法×四环境×三攻击族结果", ["方法", "环境", "攻击", "拦截", "Regret", "成本", "Brier"], rows, [2.3, 3.0, 2.5, 2.3, 2.3, 2.3, 2.3], 7.2)

    t.chapter("附录C  校准与 Regret 结果")
    calibration = read_csv(ROOT / "outputs/stage17_3_full_benchmark/calibration_results.csv")
    rows = [
        (r["method"], r["environment"], r["attack_family"], r["brier_score"][:7], r["expected_calibration_error"][:7], r["negative_log_likelihood"][:7])
        for r in calibration
    ]
    t.table("多环境概率校准结果", ["方法", "环境", "攻击", "Brier", "ECE", "NLL"], rows, [2.5, 3.2, 2.6, 2.1, 2.1, 2.1], 7.4)

    t.chapter("附录D  失败案例与 Claim 边界")
    failures = read_csv(ROOT / "outputs/stage14_paper_suite/failure_case_table.csv")
    failure_domains = {
        "7.3": "欺骗学习",
        "9.2": "可辨识性",
        "10": "推断—决策",
        "10.3": "鲁棒性",
        "12": "主动控制",
        "14": "经典规划",
    }
    rows = [(failure_domains.get(r["stage"], "综合诊断"), r["failure_case"], r["condition"], r["evidence"], r["mechanism"]) for r in failures]
    t.table("关键失败案例", ["问题域", "失败", "条件", "证据", "机制"], rows, [2.0, 3, 3, 4, 4], 7.8)
    t.section("D.1 可支持的核心 Claim")
    claims = [
        "C1：在所评估的部分环境中，联合 belief、counterfactual evidence 与 adaptive commitment 可以提高 interception。",
        "C2：检测 AUROC 与最终 interception 必须分别报告；二者可以显著背离。",
        "C3：trajectory-only detector 在可观察别名事件上受到不可消除的 Bayes error 下界。",
        "C4：多环境完整基准中的 RIDe 优势具有环境依赖性，不构成普遍最优证明。",
        "C5：MAPPO 的 500k 总预算结果证明当前实现收敛到局部策略；不能推出所有 MAPPO 都无效。",
    ]
    for claim in claims:
        t.bullet(claim)
    t.section("D.2 明确不支持的 Claim")
    rejected = [
        "RIDe 在任意环境和任意欺骗策略下都优于所有基线。",
        "规则 adaptive attacker 等价于最优对手或纳什均衡。",
        "较高 deception F1 足以保证成功拦截。",
        "增加训练步数必然解决立即 commit 或 stay collapse。",
        "Forked Routes 接口通过即可证明神经 encoder 已完成跨环境迁移。",
    ]
    for claim in rejected:
        t.bullet(claim)

    t.chapter("附录E  实验产物与复现路径")
    artifacts = [
        ("双头信念模型", "stage10-two-level/belief_model.pt", "29bef053…c883b"),
        ("概率校准器", "stage10-1-decoupled/calibrator.pt", "5e945db2…aad64"),
        ("解耦控制评估数据", "stage10-1-decoupled/evaluation_episodes.csv", "6f7e74a1…e1b8b3"),
        ("学习型控制器", "outputs/stage13_learned_controller/stage13_controller.pt", "冻结 checkpoint"),
        ("失败归因结果", "outputs/stage16_failure_attribution/", "CSV、图和报告"),
        ("多环境完整基准", "outputs/stage17_3_full_benchmark/", "36,000 episodes"),
    ]
    t.table("主要复现实验产物", ["产物", "路径", "说明/hash"], artifacts, [4, 8, 4], 8.5)
    t.p("复现时应运行测试确认环境接口，并读取完整基准目录中的 config.json 和 preserved_inputs.json；需要重新训练 MAPPO 时运行 python -m experiments.run_stage17_full_benchmark。命令名与目录名沿用代码仓库中的历史标识，仅用于定位产物，不表示论文的论证顺序。runner 支持 latest checkpoint 续跑，并在完成后自动生成 CSV、LaTeX 表和 PNG 图。")

    t.chapter("附录F  五随机种子逐条件统计")
    t.p("本附录给出多环境完整基准的 seed-level 聚合结果。每行包含同一 seed、方法、环境和攻击设置下的 100 个 episode；正文与附录B中的 mean±std 由这里的五行 seed 统计计算。保留逐 seed 数据可以检查方差、目标平衡和个别随机种子异常，避免仅凭总样本均值掩盖不稳定性。")
    seed_data = read_csv(ROOT / "outputs/stage17_3_full_benchmark/per_seed_results.csv")
    method_short = {"rule_active": "Rule", "bayesian": "Bayes", "pomcp": "POMCP", "active_inference": "AI", "mappo": "MAPPO", "ride": "RIDe"}
    environment_short = {"gridworld": "GW", "forked_routes": "Fork", "pettingzoo_deception": "PZ", "cyber_deception": "Cyber"}
    family_short = {"scripted": "script", "adaptive": "adapt", "unseen_family": "unseen"}
    seed_rows = [
        (
            r["seed"],
            method_short[r["method"]],
            environment_short[r["environment"]],
            family_short[r["attack_family"]],
            f"{float(r['interception']):.3f}",
            f"{float(r['regret']):.3f}",
            f"{float(r['observation_cost']):.3f}",
            f"{float(r['calibration']):.3f}",
        )
        for r in seed_data
    ]
    t.table(
        "多环境基准 seed-level 结果（每行100 episodes）",
        ["Seed", "方法", "环境", "攻击", "拦截", "Regret", "成本", "Brier"],
        seed_rows,
        [1.0, 2.0, 2.8, 2.2, 1.6, 1.6, 1.6, 1.6],
        6.5,
    )

    t.chapter("附录G  MAPPO 数值训练曲线")
    t.p("本附录列出五个 MAPPO seed 的每次 PPO update 指标。训练共约 500,000 timesteps；evaluation_success 使用 scripted 与 adaptive 训练条件上的固定 deterministic validation。五个 seed 最终窗口均稳定为 0.5，但 entropy、critic loss 和 explained variance 仍发生变化，说明参数在更新而策略输出落入即时承诺局部解。")
    curves = read_csv(ROOT / "outputs/stage17_3_full_benchmark/training_curves.csv")
    curve_rows = [
        (
            r["seed"],
            r["update"],
            r["timesteps"],
            f"{float(r['rollout_success']):.3f}",
            f"{float(r['evaluation_success']):.3f}",
            f"{float(r['actor_loss']):.4f}",
            f"{float(r['critic_loss']):.4f}",
            f"{float(r['entropy']):.4f}",
            f"{float(r['explained_variance']):.4f}",
        )
        for r in curves
    ]
    t.table(
        "MAPPO 五 seed 训练日志",
        ["Seed", "更新", "步数", "R成功", "V成功", "Actor", "Critic", "Entropy", "EV"],
        curve_rows,
        [0.9, 1.0, 1.7, 1.7, 1.7, 1.6, 1.6, 1.5, 1.4],
        6.2,
    )


def build():
    t = Thesis()
    add_cover(t)
    add_abstracts(t)
    add_front_matter(t)
    add_chapter1(t)
    add_chapter2(t)
    add_chapter3(t)
    add_chapter4(t)
    add_chapter5(t)
    add_chapter6(t)
    add_chapter7(t)
    add_chapter8(t)
    add_chapter9(t)
    add_chapter10(t)
    add_references(t)
    add_appendices(t)
    destination = OUTPUT / "RIDe主动反欺骗决策研究_硕士学位论文.docx"
    t.save(destination)
    metadata = {
        "title": TITLE + SUBTITLE,
        "document": str(destination),
        "figures": t.figure_number,
        "tables_and_algorithms": t.table_number,
        "source": "Frozen repository artifacts covering method development, robustness analysis, and external validation",
        "paper_writing_only": True,
    }
    (OUTPUT / "thesis_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    print(build())
