"""Render the active three-node baseline_v2 workflow as SVG and PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(19, 17))
fig.patch.set_facecolor("#f7f9fc")
ax.set_facecolor("#f7f9fc")
ax.set_xlim(0, 1800)
ax.set_ylim(1700, 0)
ax.axis("off")

COLORS = {
    "input": ("#e9f3ff", "#5b8bbd"),
    "qwen": ("#eee9ff", "#8568bd"),
    "code": ("#fff1d6", "#bd8a32"),
    "branch": ("#e7f7f5", "#4f9992"),
    "tool": ("#e3f5e9", "#55956b"),
    "feedback": ("#fff0eb", "#bf7465"),
}
INK = "#17324b"
BODY = "#3b5369"
ARROW = "#59748b"
LANES = (70, 650, 1230)
WIDTH = 500


def card(x, y, h, title, lines, kind):
    fill, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), WIDTH, h, boxstyle="round,pad=0.01,rounding_size=18",
        linewidth=2, edgecolor=edge, facecolor=fill,
    ))
    ax.text(x + 25, y + 38, title, fontsize=18, weight="bold", color=INK,
            va="center")
    ax.text(x + 25, y + 69, "\n".join(lines), fontsize=14.3,
            color=BODY, va="top", linespacing=1.4)


def arrow(start, end, label="", color=ARROW, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=20,
        linewidth=2.4, linestyle="--" if dashed else "-", color=color,
    ))
    if label:
        ax.text((start[0] + end[0]) / 2 + 8,
                (start[1] + end[1]) / 2 - 8, label,
                fontsize=12.5, color=color,
                bbox={"facecolor": "#f7f9fc", "edgecolor": "none", "pad": 2})


def route(points, label="", color=ARROW, dashed=False):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, linewidth=2.4, color=color,
            linestyle="--" if dashed else "-", solid_capstyle="round")
    arrow(points[-2], points[-1], label=label, color=color, dashed=dashed)


ax.text(900, 63, "LifeMate baseline_v2 · Workflow hiện tại",
        ha="center", fontsize=30, weight="bold", color=INK)
ax.text(900, 111, "Ba node Qwen · checklist request · sửa field cục bộ · kiểm trước và sau tool",
        ha="center", fontsize=16, color=BODY)

for x, title, sub in zip(LANES,
                         ("1 · Planner", "2 · Chọn argument", "3 · Điền và chạy"),
                         ("chọn bước tiếp", "required + x-one-of + optional",
                          "value + kiểm tra + tool")):
    ax.add_patch(FancyBboxPatch(
        (x, 155), WIDTH, 78, boxstyle="round,pad=0.01,rounding_size=15",
        linewidth=0, facecolor="#dce8f3"))
    ax.text(x + 25, 187, title, fontsize=21, weight="bold", color=INK)
    ax.text(x + 25, 218, sub, fontsize=13.5, color=BODY)

# The actual pipeline in agent_pipeline.py. Qwen calls for selectors are conditional.
card(70, 270, 175, "Input của planner", (
    "request · context · history · available_tools",
    "observations · execution_status · feedback",
    "task_plan · request_requirements"), "input")
card(70, 510, 175, "Qwen: planner", (
    "Lần đầu lập đủ goal và điều kiện",
    "Chọn call / ask / revise / done",
    "Rà goal, điều kiện và việc còn thiếu"), "qwen")
card(70, 755, 175, "Nhánh action", (
    "call → chọn một tool khả dụng",
    "ask → tra lookup và dữ kiện đã biết",
    "revise / done → sửa plan / kiểm evidence"), "branch")
card(70, 1030, 255, "Code: ask / revise / done", (
    "ask: kiểm dữ kiện và lookup trước khi hỏi",
    "revise: sửa plan; thêm goal không tốn quota",
    "done: checklist request + evidence từng goal",
    "Đạt → trả tool_calls + state_hash",
    "Chưa đạt → feedback cho planner"), "branch")

card(650, 270, 165, "Code: đọc ràng buộc schema", (
    "required được code ghép tự động",
    "Xác định field bổ sung cần model chọn"), "code")
card(650, 510, 185, "Qwen: argument selector", (
    "Chọn field bổ sung cho x-one-of",
    "và optional; không điền value/lý do",
    "Bỏ qua node nếu không có field để chọn"), "qwen")
card(650, 765, 195, "Code: ghép và kiểm field", (
    "required + selected_arguments",
    "+ intent_required_fields theo yêu cầu",
    "Kiểm tên field và đủ một nhóm x-one-of",
    "Bỏ optional không thuộc tác vụ"), "code")
card(650, 1030, 175, "Không có field?", (
    "selected_arguments=[] → arguments={}",
    "Bỏ qua value selector; đi tới validate",
    "Có field → value selector"), "branch")

card(1230, 270, 200, "Qwen: value selector", (
    "Điền arguments + value_reasons",
    "value_hints · trusted · reference_candidates",
    "Đúng đối tượng, điều kiện và schema",
    "Thiếu field → argument_selection_error"), "qwen")
card(1230, 545, 215, "Code: kiểm trước khi gọi", (
    "Đủ required / x-one-of; ID đúng collection",
    "Đủ field theo ý định · giá trị đúng nhánh",
    "Kiểm ambiguity / write đã xong trong history",
    "env.validate · chặn read/write lặp",
    "Sai → feedback; đạt → thực thi"), "code")
card(1230, 810, 145, "env.call(name, arguments)", (
    "Chỉ bước này thực thi tool",
    "và có thể đổi trạng thái"), "tool")
card(1230, 1030, 255, "Code: kiểm result và effect", (
    "Lưu result + trạng thái thành công",
    "Kiểm effect của tool được hỗ trợ",
    "Mâu thuẫn → không dùng để hoàn tất goal",
    "Tool chưa phủ: effect_verified = null",
    "Cập nhật observations → planner"), "code")

arrow((320, 445), (320, 510))
arrow((320, 685), (320, 755))
arrow((320, 930), (320, 1030), label="ask / revise / done", color="#4f9992")
route([(570, 840), (610, 840), (610, 350), (650, 350)])
arrow((900, 435), (900, 510))
arrow((900, 695), (900, 765))
arrow((900, 960), (900, 1030))
route([(1150, 1110), (1190, 1110), (1190, 370), (1230, 370)])
route([(1150, 1140), (1210, 1140), (1210, 640), (1230, 640)],
      color="#4f9992", dashed=True)
route([(1230, 320), (1174, 320), (1174, 600), (1150, 600)],
      color="#a45c88", dashed=True)
ax.text(1175, 485, "Sửa field\ntối đa 1 lần", fontsize=11.5, color="#a45c88",
        bbox={"facecolor": "#f7f9fc", "edgecolor": "none", "pad": 2})
arrow((1480, 470), (1480, 545))
arrow((1480, 760), (1480, 810))
arrow((1480, 955), (1480, 1030))

fill, edge = COLORS["feedback"]
ax.add_patch(FancyBboxPatch(
    (70, 1430), 1660, 130, boxstyle="round,pad=0.01,rounding_size=18",
    linewidth=2, edgecolor=edge, facecolor=fill))
ax.text(105, 1471, "Feedback và giới hạn", fontsize=18, weight="bold", color=INK)
ax.text(105, 1511,
        "Lỗi tool / field / value / điều kiện / done → planner. Hết budget hoặc không tiến triển → dừng có stop_reason.",
        fontsize=14.2, color=BODY)
route([(1730, 640), (1775, 640), (1775, 1495), (1730, 1495)],
      color="#bf7465")
route([(70, 1495), (38, 1495), (38, 640), (70, 640)],
      color="#bf7465", dashed=True)
route([(1480, 1285), (1480, 1600), (18, 1600), (18, 595), (70, 595)])
ax.text(900, 1642,
        "Tím: Qwen · Vàng: code · Xanh lá: tool. Thường ≤ 3 lượt Qwen/bước; sửa field có thể thêm 2 lượt.",
        ha="center", fontsize=13.2, color=BODY)

ax.text(900, 1678, "Checklist và effect checker có phạm vi hữu hạn; chưa chứng minh mọi ý định/ngữ nghĩa đều đúng.",
        ha="center", fontsize=12.3, color=BODY)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(ROOT / "workflow.svg", facecolor=fig.get_facecolor())
fig.savefig(ROOT / "workflow.png", dpi=130, facecolor=fig.get_facecolor())
plt.close(fig)
