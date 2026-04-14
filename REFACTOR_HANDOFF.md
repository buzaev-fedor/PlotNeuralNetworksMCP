# Refactor Handoff — для следующего агента

> Этот документ — handoff после семи фаз рефактора `plot_nn_mcp`. Цель — позволить следующему агенту продолжить работу автономно, не перечитывая всю историю переписки. Документ читается линейно.

---

## 1. Executive Summary

### Что было сделано

Переписан pipeline рендеринга диаграмм с **монолитного однопроходного транслятора** (1500+ строк, 50+ `isinstance` веток в одной функции, residual зашит в каждый блок отдельно) на **трёхслойную архитектуру**:

```
DSL Layer dataclass  →  IR (DAG)  →  emit_tikz (layout + render)
```

Каждый слой имеет тестируемые контракты. Семь фаз закончены, **default render path = `use_ir=True`**, все 25 baseline-архитектур из [examples/generate_all.py](examples/generate_all.py) рендерятся через новый pipeline, **590 тестов зелёные**.

Закрыты структурно (то есть не патчем, а невозможностью повторения):
- Mixtral MoE без residual (теперь `IRResidualOp` обёртка едина для attn и MoE — забыть skip нельзя)
- Input arrow цепляется к SectionHeader (теперь `_first_semantic_node()` пропускает structural anchors)
- DeBERTa double `{Output}` (теперь `_redundant()` в emit_tikz суппрессирует дубль метки)
- Дробные cm `1.8666666666666667cm` (теперь `round(w, 2)` в emit_tikz, инвариант проверяется тестом)
- Цветовая семантика плавала между блоками (теперь `Role` enum, `resolve_fill` единая точка резолва)

### Что осталось

Разделено на 12 задач, ранжированных по выгоде. Самое выгодное:

1. **Декомпозиция compact-lowering для LSTM/GRU/Mamba/Generator/Discriminator/UNetBlock** — сейчас они рендерятся как один box (визуальная регрессия в 3 baseline).
2. **`horizontal` / `unet` layouts через IR** — сейчас обходят IR полностью, теряя все фиксы.
3. **Удаление мёртвого legacy-кода `_render_vertical`** (после задач 1-2 он становится недостижим, ~1500 строк).
4. **Cleanup мёртвого кода в преамбуле** (`smallblock`, `clrbackground`, `clrdense`).
5. **`EncoderDecoder` / `SideBySide` / `UNetLevel` / `Bottleneck` через `IRParallelOp`** — для пользовательских presets (не используются в baseline, но могут быть в чужом коде).
6. **Skip-arrow xshift из bbox группы** — сейчас магические 2.0-2.6cm, может пересекаться с группой.
7. **Документация** — README, CHANGELOG, MCP schema.
8. **Property-based тесты** через Hypothesis.

### Почему это handoff, а не «сделай ещё»

Каждая из задач — несколько часов работы с обязательным циклом «тест → имплементация → golden update → ревью». Документ структурирован так, чтобы агент мог взять одну задачу, выполнить её до мержа, и перейти к следующей. **Нельзя выполнять задачи параллельно** — большинство трогают одни и те же файлы.

---

## 2. Архитектурный recap

### Три слоя и их контракты

#### Слой 1: DSL (`src/plot_nn_mcp/dsl.py`)

Публичный API. Pydantic-style dataclass'ы (`Embedding`, `TransformerBlock`, `MoELayer`, ...) и контейнер `Architecture`:

```python
arch = Architecture(name, theme="modern", layout="vertical")
arch.add(Embedding(d_model=768, label="Token Emb"))
arch.add(TransformerBlock(attention="self", norm="post_ln"))
arch.add(ClassificationHead(label="[CLS]"))
tex = arch.render(show_n=4, use_ir=True)  # use_ir=True is now default
```

**Инвариант:** публичный API не должен меняться. Все 47 layer-типов dataclass остаются. Новые типы добавляются через `register(MyLayer, my_lowering)` (см. [lowering.py](src/plot_nn_mcp/lowering.py)).

#### Слой 2: IR ([src/plot_nn_mcp/ir.py](src/plot_nn_mcp/ir.py))

Промежуточное представление в виде DAG. Ключевые типы:

- **`IRNode(id, role, label, shape, size_hint, dim, meta)`** — вершина графа. `shape ∈ {"block", "circle", "anchor", "section_rule"}`. `role: Role` определяет цвет.
- **`IREdge(src, dst, kind)`** — ребро. `kind ∈ {"data", "skip", "branch", "merge"}`.
- **`IRGroup(id, children, title, repeat_count)`** — группа узлов для `fit=` рамки и ×N бейджа.
- **`IRGraph(nodes, edges, groups, order, input_node, output_node, title)`** — весь граф.

Композиционные операторы (Phase 4):
- **`IRBlockOp`** — лист, превращается в один `IRNode`.
- **`IRSequenceOp(ops)`** — цепочка `data` рёбер.
- **`IRResidualOp(body)`** — entry → body → add ← skip(entry → add). **Именно эта обёртка делает Mixtral баг невозможным.**
- **`IRParallelOp(branches, merge)`** — split → параллельные ветви → optional merge.
- **`IRCustomOp(nodes, edges, entry_id, exit_id)`** — escape hatch для legacy renderers.

`IRBuilder` имеет fluent API:
```python
b = IRBuilder(title="Mini-BERT")
b.add_block(Role.EMBED, "Embedding", dim=768)
b.add_op(IRResidualOp(body=[
    IRBlockOp(Role.NORM, "LayerNorm"),
    IRBlockOp(Role.ATTENTION, "Self-Attn"),
]))
graph = b.build()  # set_input/set_output called automatically if missing
```

**Контракт IR:** `input_node` — первый **семантический** (не `section_rule`/`anchor`) узел. `output_node` — последний узел в `order` ИЛИ явно установленный через `set_output`.

#### Слой 3: Render ([src/plot_nn_mcp/render.py](src/plot_nn_mcp/render.py))

Функция `emit_tikz(graph: IRGraph, theme: Theme) -> str`. Пайплайн:

1. `_normalize_widths_within_runs(graph)` — выравнивает ширины подряд идущих блоков одной роли (Phase 5 фикс stem-conv jitter).
2. Рендер преамбулы, цветов, начала tikzpicture.
3. Для каждого узла в `graph.order` — `_emit_node(node, prev_id, is_first)`. Округление `round(w, 2)` происходит здесь.
4. Все рёбра одним проходом — `data` → `flat_arrow`, `skip` → `flat_skip_arrow`. Ветви `branch`/`merge` сейчас тоже как `data` (TODO).
5. Input/Output стрелки с suppression если узел уже имеет label "Input"/"Output".
6. Group frames: `\node[draw=clrgroup_frame, ..., fit=(child1) (child2) ...]` + `×N` бейдж справа.
7. Subtitle с `arch.title`.
8. `\end{tikzpicture}\end{document}`.

**Контракт render:** не должен возвращать невалидный LaTeX. Не должен эмитить дробные cm с >2 знаками. Для каждого `IRGraph` с `input_node` / `output_node` должен быть ровно один `{Input}` и один `{Output}` маркер (с учётом suppression).

### Lowering ([src/plot_nn_mcp/lowering.py](src/plot_nn_mcp/lowering.py))

Мост DSL → IR. Dispatch table `_LOWERING: dict[type, callable]`. Регистрация:
```python
register(dsl.MyLayer, my_layer_to_ir)
```

`architecture_to_ir(arch, show_n=4)` — основная функция. Делает:
1. Запускает `_detect_groups(arch.layers)` (legacy функция в dsl.py).
2. Для каждого видимого слоя (rep_index < show_n) вызывает `layer_to_ir(layer)` → `IROp` → `builder.add_op(op)`.
3. Собирает `group_node_ids` для последующих `IRGroup` с fit-рамкой.
4. Pin output до badge анкоров.
5. Возвращает `builder.build()`.

`coverage()` и `LEGACY_ONLY` отслеживают migration progress (тест `test_coverage_progress` ловит регрессии).

### Theme ([src/plot_nn_mcp/themes.py](src/plot_nn_mcp/themes.py))

```python
class Role(str, Enum):
    ATTENTION, ATTENTION_ALT, FFN, NORM, EMBED, RESIDUAL,
    OUTPUT, DENSE, SPECTRAL, PHYSICS
```

`resolve_fill(role_or_str) → str` — единая точка резолва имени цвета. Принимает либо `Role`, либо строку (для обратной совместимости с legacy callsites).

`Theme` — frozen dataclass с hex-полями (`attention`, `ffn`, ...). Конвертится в TikZ через `theme_to_tikz_colors(theme)`.

### Поток данных, end-to-end

```
arch = Architecture("BERT")
arch.add(Embedding(...))
arch.add(TransformerBlock(...))
...
arch.render(show_n=3, use_ir=True)
    │
    ├─→ get_theme("modern")  # → Theme dataclass
    │
    ├─→ can_lower_architecture(arch)  # checks all layer types in _LOWERING
    │     False → fall back to _render_vertical (legacy)
    │     True ↓
    │
    ├─→ architecture_to_ir(arch, show_n=3)
    │     │
    │     ├─ _detect_groups(arch.layers)   # find ×N patterns
    │     ├─ for each visible layer:
    │     │     layer_to_ir(layer) → IROp
    │     │     builder.add_op(op) → adds nodes/edges, advances cursor
    │     ├─ pin builder.set_output(builder.cursor)
    │     ├─ for each group with len > 1: new_group, attach children
    │     └─ return builder.build()
    │
    └─→ emit_tikz(graph, theme) → str
          │
          ├─ _normalize_widths_within_runs(graph)
          ├─ flat_head + flat_colors + flat_begin
          ├─ for each node: _emit_node(...) → flat_block / flat_op_circle / section_rule
          ├─ for each edge: flat_arrow / flat_skip_arrow
          ├─ Input/Output arrows (suppress if redundant)
          ├─ for each group: \node[..., fit=...] + \times{N} badge
          ├─ subtitle
          └─ flat_end
```

---

## 3. Test infrastructure — критическая для безопасной работы

Тестовая инфраструктура — **главное достижение Phase 1**, без которого весь рефактор был бы вслепую. **Не сломайте её**.

### Golden snapshots ([tests/golden/](tests/golden/))

25 файлов `.tex`, по одному на baseline-архитектуру. Сравниваются побайтово в `test_snapshot_matches_golden`. Регенерация:

```bash
UPDATE_GOLDEN=1 pytest tests/test_golden.py::test_snapshot_matches_golden
```

**Только** после намеренного изменения output. После регенерации обязательно посмотреть `git diff tests/golden/` глазами — все ли изменения ожидаемы.

### Структурные инварианты в [tests/test_golden.py](tests/test_golden.py)

Проверяются на каждой из 25 baseline-архитектур:
- `test_latex_balanced` — `\begin{document}/\end{document}` и `\begin{tikzpicture}/\end{tikzpicture}` парные.
- `test_exactly_one_input_output_label` — ≤1 `{Input}` и ≤1 `{Output}`.
- `test_skip_arrow_has_matching_add_circle` — каждый skiparrow завершается на add-circle.
- `test_no_fractional_cm_baseline` — **глобально** ноль дробных cm (был allow-list во время transition).
- `test_structural_document` — есть `\documentclass`, `\usetikzlibrary{positioning`, заканчивается `\end{document}`.

### IR-уровневые инварианты ([tests/test_ir.py](tests/test_ir.py))

- `test_input_defaults_to_first_semantic_node` — `IRBuilder` пропускает `section_rule` при auto-input.
- `test_residual_op_lowering_generates_skip_edge` — каждый `IRResidualOp` даёт ровно один skip edge.
- `test_sequence_of_two_residuals_fixes_mixtral_pattern` — два sequential residual = 2 skip edges. **Доказательство Mixtral fix.**
- `test_parallel_op_preserves_branch_edges` — `IRParallelOp` сохраняет `branch` тип edges.
- `test_duplicate_node_id_rejected` — IRGraph не позволяет дублировать id.

### End-to-end через render ([tests/test_render.py](tests/test_render.py))

- `test_emit_residual_has_skiparrow_and_add` — IR с residual → TikZ с skiparrow и `{+}`.
- `test_no_fractional_cm_after_rounding` — проверяет округление в emitter.
- `test_input_skips_section_rule_node` — Input arrow не цепляется к section_rule.

### Bug-fix proofs ([tests/test_ir_path_fixes.py](tests/test_ir_path_fixes.py))

19 тестов которые **сравнивают legacy vs IR** для каждого исправленного бага:
- `test_mixtral_ir_path_has_moe_residual` — IR > legacy по skip count.
- `test_deberta_ir_path_no_double_output` — IR=1, legacy=2.
- `test_efficientnet_ir_path_no_fractional_cm` / `test_fno_2d_...` — IR=0.
- `test_mixtral_ir_path_has_pre_moe_norm` — IR > legacy LN count.
- `test_ir_path_input_attaches_to_real_block` — для всех 12 IR-ready (legacy не проходит).
- `test_ir_path_produces_valid_latex[idx]` — все 25 архитектур, валидный LaTeX.
- `test_ir_path_no_fractional_cm_anywhere[idx]` — все 25, ноль дробных cm.
- `test_ir_path_at_most_one_output_label[idx]` — все 25, ≤1 Output.
- `test_legacy_path_still_invokable` — legacy доступен через `use_ir=False`.

### Lowering invariants ([tests/test_lowering.py](tests/test_lowering.py))

- `test_coverage_progress` — пинит количество мигрированных типов. **Поднимайте порог по мере миграции.** Сейчас `>= 31` (фактически 39/47). После задачи #1 поднять до 45+.
- `test_unregistered_layer_raises` — добавил тип без регистрации → понятная ошибка.
- `test_*_has_residual` — структурные проверки конкретных lowering функций.

### Запуск тестов

```bash
pytest tests/                                # все 590
pytest tests/test_lowering.py -v             # с verbose
pytest tests/test_golden.py::test_snapshot_matches_golden -q  # только snapshots
UPDATE_GOLDEN=1 pytest tests/test_golden.py::test_snapshot_matches_golden -q  # update
pytest tests/ -q --tb=line                   # короткий traceback
pytest tests/ -k "lstm" -v                   # фильтр по имени
```

**Перед каждым коммитом:** `pytest tests/ -q` → должно быть «X passed, 0 failed».

---

## 4. Детальный план задач

Задачи в порядке убывания выгоды/возрастания сложности. Каждая задача автономна — можно мержить отдельно.

---

### Задача 1: Декомпозиция LSTM / GRU / Mamba

**Контекст.** Сейчас [lowering.py:lstm_block_compact_to_ir](src/plot_nn_mcp/lowering.py) рендерит `LSTMBlock` как один box `"Bi-LSTM (256)"` вместо детализированных гейтов. Аналогично для GRU и Mamba. Затронуты baseline-архитектуры:
- `01_lstm_ner` — Bidirectional LSTM
- `02_gru_translation` — GRU encoder/decoder
- `03_seq2seq_attention` — LSTM с attention
- В legacy renderer ([dsl.py:_render_lstm_block](src/plot_nn_mcp/dsl.py) ~ строки 765-870) гейты рисовались параллельным паттерном `[forget_gate, input_gate, cell_update, output_gate]` с `\sigma`/`\tanh` метками.

**Цель.** Lowering функции `lstm_block_to_ir`, `gru_block_to_ir`, `mamba_block_to_ir` возвращают IR который воспроизводит детализированный гейт-вид.

**Файлы:**
- `src/plot_nn_mcp/lowering.py` — заменить compact функции
- `tests/test_lowering.py` — добавить тесты на структуру
- `tests/golden/` — обновить через `UPDATE_GOLDEN=1`

**Подход.**

```python
def lstm_block_to_ir(block: dsl.LSTMBlock) -> IROp:
    if block.style == "compact":
        return lstm_block_compact_to_ir(block)  # leave as-is for compact mode

    # Default "gates" style: 4 gates in parallel, then cell update + output.
    # forget gate (σ), input gate (σ), cell candidate (tanh), output gate (σ)
    gates = IRParallelOp(
        branches=[
            IRBlockOp(Role.NORM, r"$f_t = \sigma$", size_hint=(2.0, 0.6)),
            IRBlockOp(Role.NORM, r"$i_t = \sigma$", size_hint=(2.0, 0.6)),
            IRBlockOp(Role.FFN, r"$\tilde c_t = \tanh$", size_hint=(2.4, 0.6)),
            IRBlockOp(Role.NORM, r"$o_t = \sigma$", size_hint=(2.0, 0.6)),
        ],
        merge="concat",
    )
    cell = IRBlockOp(Role.RESIDUAL, r"$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$",
                    size_hint=(4.6, 0.7))
    out = IRBlockOp(Role.OUTPUT, r"$h_t = o_t \odot \tanh(c_t)$",
                    size_hint=(4.6, 0.7))
    return IRSequenceOp(ops=[gates, cell, out])
```

**Гочи:**
1. `IRParallelOp` сейчас лоуэрится в IR с `kind="branch"` edges, но `emit_tikz` рендерит их как обычные `data` arrows (TODO). Чтобы реально нарисовать ветви бок-о-бок, надо **сначала** добавить параллельный layout в `emit_tikz` (см. задачу 2 ниже — она их использует тоже). Можно: a) сделать compact-параллель «псевдо» (4 узла подряд через `IRSequenceOp` с разделителями) b) сделать настоящий параллельный layout.
2. LSTMBlock имеет поле `bidirectional`, `style ∈ {"gates", "olah", "compact"}`, `unroll`. Olah — это горизонтальный belt, нужен `layout="horizontal"` или особый IR. Compact уже сделан. **Минимально:** реализовать только `style == "gates"`, остальные — fall back на compact.
3. Mamba — Selective SSM. В legacy [dsl.py:_render_mamba_block](src/plot_nn_mcp/dsl.py) рисуется как Linear → SSM (с timestep arrow) → Linear. Можно сделать `IRSequenceOp` из 3 блоков.

**Тесты добавить:**
```python
def test_lstm_gates_style_has_four_gate_blocks():
    op = layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="gates"))
    # IRSequenceOp([IRParallelOp([...4 gates...]), cell, out])
    assert isinstance(op, IRSequenceOp)
    parallel = op.ops[0]
    assert isinstance(parallel, IRParallelOp)
    assert len(parallel.branches) == 4

def test_lstm_compact_still_one_box():
    op = layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="compact"))
    assert isinstance(op, IRBlockOp)

def test_lstm_baseline_renders_with_gates():
    # Take 01_lstm_ner from baselines; check rendered .tex contains gate symbols
    arch = baselines["01_lstm_ner"]
    tex = arch.render(use_ir=True)
    assert r"\sigma" in tex
    assert r"\tanh" in tex
```

**Критерий приёмки:**
- `pytest tests/` → 590+ зелёные.
- `python examples/generate_all.py` → визуально (если есть LaTeX) `01_lstm_ner.pdf` показывает гейты.
- `tests/golden/01_lstm_ner.tex` обновлён через `UPDATE_GOLDEN=1`, diff содержит ожидаемые `\sigma`/`\tanh`.
- `coverage()` показывает migrated >= 39 (без регрессии — `LSTMBlock` уже был мигрирован compact-стилем).

**Оценка:** ~3-4 часа на LSTM, ~1 час каждый на GRU/Mamba (структурно похожи).

---

### Задача 2: Параллельный layout для `IRParallelOp`

**Контекст.** `IRParallelOp` сейчас лоуэрится в IR корректно (см. `lower()` в [ir.py](src/plot_nn_mcp/ir.py)), но `emit_tikz` рендерит ветви **последовательно** через `above_of`. Это значит, что YOLOv8 detect-heads (если их сделать через `IRParallelOp`) визуально останутся столбцом, а не тремя параллельными выходами.

**Цель.** В `emit_tikz` детектить группы веток (рёбра с `kind="branch"` от одного source) и размещать их горизонтально через `right=` / `left=` относительно split-anchor.

**Файлы:**
- `src/plot_nn_mcp/render.py` — основная работа
- `tests/test_render.py` — новые тесты
- Возможно: `src/plot_nn_mcp/layout.py` (новый файл, если выделять layout pass)

**Подход.**

В `emit_tikz`, перед эмиссией node'ов, пройти по edges, построить map `split_node → list[branch_dst]`. Для каждого split:
1. Эмитнуть split-anchor сам.
2. Эмитнуть первую ветку как `above_of=split` (центр).
3. Эмитнуть остальные ветки как `right=of branch1` или `left=of branch1` (симметрично распределить).
4. Внутри каждой ветви — обычная вертикальная цепочка через свои `data` edges.
5. Если есть `merge` — эмитнуть merge-node с `above of branch_center` и стрелки от каждого `branch.exit → merge`.

```python
def _detect_split_groups(graph: IRGraph) -> dict[str, list[str]]:
    """Map split-anchor id → list of first-node ids of each branch."""
    out: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind == "branch":
            out.setdefault(edge.src, []).append(edge.dst)
    return out
```

**Гочи:**
1. **Mixed positioning.** Сейчас `_emit_node` использует только `above_of=prev_id`. Для веток нужен `right_of` / `left_of`. Расширить функцию `_emit_node` чтобы принимала explicit position override.
2. **Edge rendering.** Для split: arrow от split → каждая branch.entry. Для merge: arrow от каждой branch.exit → merge. Эти edges уже в графе с правильными kinds (`branch`/`merge`), не надо угадывать.
3. **Вложенные параллели.** `IRParallelOp` внутри `IRParallelOp` (например, multi-head attention с per-head FFN). Layout solver должен быть рекурсивным. Минимально — поддержать только верхний уровень, throw NotImplementedError для вложенных.
4. **Y-выравнивание.** Если ветки разной высоты (one branch has 3 nodes, another 5), merge-node должен быть выше самой высокой ветки. Можно использовать TikZ `let` с max() или просто использовать самую длинную для расчёта.

**Тесты:**
```python
def test_parallel_op_renders_branches_horizontally():
    b = IRBuilder()
    src = b.add_block(Role.ATTENTION, "Backbone")
    b.add_op(IRParallelOp(branches=[
        IRBlockOp(Role.OUTPUT, "Head A"),
        IRBlockOp(Role.OUTPUT, "Head B"),
        IRBlockOp(Role.OUTPUT, "Head C"),
    ], merge="none"))
    tex = emit_tikz(b.build(), get_theme("modern"))
    # Two branches must use right=/left= positioning, not all above
    assert "right=" in tex or "left=" in tex
    # All three head names present
    assert all(h in tex for h in ["Head A", "Head B", "Head C"])

def test_parallel_op_with_merge_emits_merge_arrows():
    ...
    assert tex.count(r"merge") >= 3  # 3 branch->merge arrows
```

**Критерий приёмки:**
- `pytest tests/` → all green.
- Есть тест `test_yolov8_detect_heads_are_parallel` который строит YOLO-подобную архитектуру и проверяет horizontal layout.
- Если задача 1 уже сделана, LSTM gates style теперь рисуются параллельно (4 гейта бок о бок) вместо серии.

**Оценка:** ~4-5 часов. Это самая «архитектурно» сложная оставшаяся задача — нужна работа с layout-solver.

---

### Задача 3: `horizontal` и `unet` layouts через IR

**Контекст.** Сейчас в [dsl.py:Architecture.render](src/plot_nn_mcp/dsl.py):
```python
if use_ir and self.layout == "vertical":
    ...IR path...
if self.layout == "horizontal":
    return _render_horizontal(...)  # legacy
if self.layout == "unet":
    return _render_unet(...)        # legacy
```

То есть архитектуры с `layout="horizontal"` или `"unet"` **полностью обходят IR** — все фиксы (round cm, suppress double Output, MoE residual) их не касаются. Никакая baseline их не использует, но любой пользовательский preset легко может.

**Цель.** Расширить IR-pipeline на `horizontal` и `unet` layouts, удалить вызовы `_render_horizontal` и `_render_unet` (~600 строк legacy).

**Файлы:**
- `src/plot_nn_mcp/render.py` — добавить `emit_tikz_horizontal` и `emit_tikz_unet` или параметр `layout` в общий emitter
- `src/plot_nn_mcp/dsl.py` — переключатель `Architecture.render`
- `src/plot_nn_mcp/lowering.py` — `architecture_to_ir` принимает layout

**Подход.**

```python
def emit_tikz(graph: IRGraph, theme: Theme,
              layout: Literal["vertical", "horizontal", "unet"] = "vertical") -> str:
    if layout == "vertical":
        return _emit_vertical(graph, theme)
    if layout == "horizontal":
        return _emit_horizontal(graph, theme)
    if layout == "unet":
        return _emit_unet(graph, theme)
    raise ValueError(f"unknown layout: {layout}")
```

`_emit_horizontal` — `west=...` / `east=...` вместо `above=...` / `below=...`. Skip arrows идут сверху/снизу вместо влево/вправо.

`_emit_unet` сложнее — это U-shape: encoder сверху-вниз слева, bottleneck в центре, decoder снизу-вверх справа, skip connections horizontal-через. В legacy [dsl.py:_render_unet](src/plot_nn_mcp/dsl.py) используется `UNetLevel` и `Bottleneck` dataclass'ы. Lowering для них уже надо делать (см. задачу 5).

**Гочи:**
1. **Horizontal flow direction.** В horizontal Input идёт слева, Output справа. `flat_arrow` должен использовать `east → west` вместо `north → south`. Уже есть `flat_arrow_h` в [flat_renderer.py](src/plot_nn_mcp/flat_renderer.py).
2. **U-Net требует UNetLevel и Bottleneck в IR.** Сейчас они в `LEGACY_ONLY`. Сначала задача 5 (или хотя бы lowering для них), потом эта.
3. **Group frames в horizontal.** Bbox `fit=` работает в любом направлении, но `\times{N}` бейдж должен идти в правильную сторону (под/над, не справа).
4. **Subtitle position.** В horizontal subtitle логичнее справа или сверху, не снизу.

**Тесты:**
```python
def test_horizontal_layout_uses_east_positioning():
    arch = Architecture("test", layout="horizontal")
    arch.add(Embedding(d_model=128))
    arch.add(ConvBlock(filters=64))
    arch.add(ClassificationHead())
    tex = arch.render()
    assert "east=" in tex or "right=" in tex
    assert "above=" not in tex.replace("above=0pt", "")  # only relative-east

def test_unet_layout_produces_skip_connections():
    ...

def test_horizontal_baseline_no_fractional_cm():
    # Pick any baseline, switch layout=horizontal, render, check
    ...
```

**Критерий приёмки:**
- `pytest tests/` → all green.
- Тест строит арх с `layout="horizontal"`, проверяет валидный TikZ + ноль дробных cm.
- Тест строит арх с `layout="unet"`, проверяет валидный TikZ.
- Можно (опционально) удалить `_render_horizontal` и `_render_unet` из dsl.py.

**Оценка:** ~5-6 часов. Horizontal — относительно простой (~2 часа), UNet — большой (~4 часа из-за U-shape геометрии).

---

### Задача 4: Skip-arrow xshift из bbox группы

**Контекст.** [flat_renderer.py:flat_skip_arrow](src/plot_nn_mcp/flat_renderer.py) принимает `xshift=2.2` cm по умолчанию. В IR-pipeline передаётся либо это default'ное значение, либо magic-numbered constants. Если в группе блок шире 4.4cm (после `_normalize_widths_within_runs`), skip-arrow упрётся в правую границу группы.

**Цель.** В layout pass посчитать максимальную ширину блоков в группе, передать в `emit_tikz` как `safe_xshift = max_width / 2 + 0.4`.

**Файлы:**
- `src/plot_nn_mcp/render.py` — основная работа в `_emit_node` для skip edges
- `src/plot_nn_mcp/lowering.py` — возможно, аннотировать residuals с `meta["safe_xshift"]`

**Подход.**

Способ 1: статический pass перед эмиссией.

```python
def _compute_safe_xshifts(graph: IRGraph) -> dict[int, float]:
    """For each skip edge, compute xshift > half the widest block on its path."""
    out: dict[int, float] = {}
    for i, edge in enumerate(graph.edges):
        if edge.kind != "skip":
            continue
        # Find all nodes between src and dst in graph.order
        src_idx = graph.order.index(edge.src)
        dst_idx = graph.order.index(edge.dst)
        path_nodes = graph.order[src_idx+1:dst_idx+1]
        max_w = max(
            (graph.nodes[n].size_hint[0] for n in path_nodes
             if graph.nodes[n].size_hint),
            default=3.8,
        )
        out[i] = max_w / 2 + 0.4
    return out
```

Затем в emit_tikz при рендере skip edge:
```python
xshift = safe_xshifts.get(edge_idx, 2.2)
parts.append(flat_skip_arrow(edge.src, edge.dst, xshift=round(xshift, 2)))
```

**Гочи:**
1. **graph.order может быть не топологический.** Сейчас он insertion-order. Для большинства IR это OK (insertion идёт линейно через builder), но для сложных случаев надо проверить.
2. **Округление.** `xshift` тоже должен идти через `round(_, 2)` чтобы не получить дробные cm.
3. **Direction.** `flat_skip_arrow` поддерживает `direction="left"|"right"`. Сейчас IR всегда использует "right". Если когда-нибудь добавите left-skip — формула та же, но знак разный.

**Тесты:**
```python
def test_skip_xshift_scales_with_block_width():
    """Wide block → skip-arrow goes wider to clear it."""
    b = IRBuilder()
    b.add_op(IRResidualOp(body=[
        IRBlockOp(Role.NORM, "wide", size_hint=(6.0, 0.85)),  # very wide
        IRBlockOp(Role.ATTENTION, "x"),
    ]))
    tex = emit_tikz(b.build(), get_theme("modern"))
    # Skip arrow should have xshift > 3.0 (= 6.0/2 + safety)
    import re
    matches = re.findall(r"\+\+\(([\d.]+),0\)", tex)
    assert all(float(m) >= 3.0 for m in matches)

def test_no_skip_arrow_overlaps_group_frame():
    """Smoke: for all baselines, no skip-arrow xshift < width/2."""
    ...
```

**Критерий приёмки:**
- `pytest tests/` → all green.
- Регенерируйте golden, посмотрите diff: некоторые xshift'ы изменятся (стали больше для широких блоков).

**Оценка:** ~2 часа.

---

### Задача 5: `EncoderDecoder`, `SideBySide`, `BidirectionalFlow` через IRParallelOp

**Контекст.** Эти типы в `LEGACY_ONLY`. Если пользователь использует их в своей архитектуре, `can_lower_architecture()` вернёт False → fall back на legacy (теряются все фиксы).

**Цель.** Добавить lowering, разблокировать пользовательские architectures.

**Файлы:**
- `src/plot_nn_mcp/lowering.py`
- `tests/test_lowering.py`

**Подход для каждого:**

#### `SideBySide(left=[...], right=[...], left_label, right_label)`

```python
def side_by_side_to_ir(layer: dsl.SideBySide) -> IROp:
    left_seq = IRSequenceOp(ops=[layer_to_ir(l) for l in layer.left])
    right_seq = IRSequenceOp(ops=[layer_to_ir(r) for r in layer.right])
    return IRParallelOp(branches=[left_seq, right_seq], merge="none")
```

Зависит от задачи 2 (параллельный layout) для красивой отрисовки, иначе будет визуально как столбцом.

#### `BidirectionalFlow(forward=[...], backward=[...])`

Аналогично SideBySide, но добавить data-edge от forward.exit к backward.entry в обратную сторону. Понадобится `kind="data"` ребро с `style={"reverse": True}` и handling в emitter.

#### `EncoderDecoder(encoder=[...], decoder=[...], cross_attention=...)`

Самый сложный. Это encoder-stack слева, decoder-stack справа, и cross-attention edges от encoder layers → decoder layers.

```python
def encoder_decoder_to_ir(layer: dsl.EncoderDecoder) -> IROp:
    # Build encoder + decoder as sub-graphs via IRCustomOp escape
    enc_builder = IRBuilder()
    for l in layer.encoder:
        enc_builder.add_op(layer_to_ir(l))
    enc_graph = enc_builder.build()

    dec_builder = IRBuilder()
    for l in layer.decoder:
        dec_builder.add_op(layer_to_ir(l))
    dec_graph = dec_builder.build()

    # Wrap as IRCustomOp; emit_tikz handles them as parallel + cross arrows
    custom_nodes = list(enc_graph.nodes.values()) + list(dec_graph.nodes.values())
    custom_edges = list(enc_graph.edges) + list(dec_graph.edges)
    # Add cross-attention edges
    if layer.cross_attention == "all":
        for enc_node in enc_graph.nodes:
            ...add edge to each dec_node...
    return IRCustomOp(nodes=custom_nodes, edges=custom_edges,
                      entry_id=enc_graph.input_node, exit_id=dec_graph.output_node)
```

**Гочи:**
1. **ID коллизии.** Каждый builder создаёт свои `n_1`, `n_2`. Если использовать два builder'а, id могут конфликтовать. Нужен prefix (`enc_n_1`, `dec_n_1`).
2. **`cross_attention` modes.** "all" — каждый dec_layer ← каждый enc_node. "last" — только last enc → first dec. "none" — без edges.
3. **EncoderDecoder vs SideBySide** — отличаются наличием cross-edges. Можно общую базу через IRParallelOp + явные cross-edges.

**Тесты:**
```python
def test_encoder_decoder_lowers_with_cross_attention():
    layer = dsl.EncoderDecoder(
        encoder=[dsl.TransformerBlock(), dsl.TransformerBlock()],
        decoder=[dsl.TransformerBlock(), dsl.TransformerBlock()],
        cross_attention="all",
    )
    op = layer_to_ir(layer)
    # IRCustomOp with extra cross-edges
    assert isinstance(op, IRCustomOp)
    cross = [e for e in op.edges if e.style.get("kind") == "cross"]
    assert len(cross) == 4  # 2 enc × 2 dec
```

**Критерий приёмки:**
- `coverage()` — `LEGACY_ONLY` сократился (минус EncoderDecoder/SideBySide/BidirectionalFlow).
- Тесты на каждый тип.
- Интеграционный: пользовательский preset с EncoderDecoder рендерится через IR без fallback.

**Оценка:** SideBySide ~1 час, BidirectionalFlow ~2 часа, EncoderDecoder ~4-5 часов.

---

### Задача 6: `UNetLevel`, `Bottleneck`, `UNetBlock`, `ForkLoss`, `DetailPanel`, `SPINNBlock`

**Контекст.** Оставшиеся 6 типов в `LEGACY_ONLY`. Используются редко в пользовательских presets.

**Цель.** Lowering для каждого.

**Файлы:** `src/plot_nn_mcp/lowering.py`, `tests/test_lowering.py`.

**Подход:**

- **`UNetLevel`** — encoder + decoder pair с skip между ними. `IRParallelOp(branches=[enc, dec])` + skip edge.
- **`Bottleneck`** — узкое место в U-Net, `IRBlockOp(Role.SPECTRAL)` — leaf.
- **`UNetBlock`** — общий U-Net фрагмент, `IRSequenceOp([conv, conv, pool])`.
- **`ForkLoss`** — multi-output losses, `IRParallelOp(branches=[loss_a, loss_b])`.
- **`DetailPanel`** — вспомогательная панель с описанием. Можно как `IRBlockOp` с большим `size_hint` или вообще `meta={"panel": True}` для special rendering.
- **`SPINNBlock`** — three-zone (Buffer, Stack, Tracker). `IRParallelOp` из 3 ветвей.

**Зависимости:** задача 2 (параллельный layout) — желательна для красивой отрисовки.

**Оценка:** ~1-2 часа на каждый, итого ~6-12 часов.

---

### Задача 7: Удаление `_render_vertical` (legacy)

**Контекст.** После задач 1+5+6, `can_lower_architecture()` вернёт True для **всех** возможных архитектур. Тогда ветка `_render_vertical(...)` в `Architecture.render` становится недостижимой (только если пользователь явно `use_ir=False`).

**Цель.** Удалить ~1500 строк legacy кода.

**Файлы:**
- `src/plot_nn_mcp/dsl.py` — большая часть удалится
- `src/plot_nn_mcp/flat_renderer.py` — возможно, неиспользуемые helpers

**Подход.**

1. Сначала пометить `Architecture.render(use_ir=False)` как deprecated (warning).
2. Переждать одну версию (с warning).
3. Затем удалить:
   - `_render_vertical`, `_render_horizontal`, `_render_unet`
   - `_render_transformer_block` (107 строк)
   - `_render_lstm_block`, `_render_gru_block`, `_render_mamba_block`
   - `_render_moe`, `_render_unet_level` и т.д.
   - `_attention_label`, `_ffn_label` дубли
   - Параметр `use_ir` из `Architecture.render` (default behavior)
4. Также удалить из тестов: `test_legacy_path_still_invokable` (legacy больше не существует).

**Гочи:**
1. **Legacy всё ещё нужен** для тех типов, которые **не** мигрированы. Если в `LEGACY_ONLY` остаются типы — этот шаг **нельзя** делать.
2. **Документация ссылается на legacy.** Поищите упоминания `_render_vertical` в [REFACTOR_PLAN.md](REFACTOR_PLAN.md) и других doc файлах.
3. **`_detect_groups` нужен IR pipeline'у** — он используется в `architecture_to_ir`. Не удаляйте его.

**Тесты:**
- `test_coverage_progress` — обновить порог до `migrated == total` и `legacy_only == 0`.
- Удалить `test_legacy_path_still_invokable`.
- Все остальные тесты должны остаться зелёными.

**Критерий приёмки:**
- `wc -l src/plot_nn_mcp/dsl.py` → < 1500 строк (было 2867).
- `LEGACY_ONLY = frozenset()` — пустое.
- `pytest tests/` — все зелёные.

**Оценка:** ~3 часа после задач 1-6.

---

### Задача 8: Cleanup мёртвого кода в преамбуле

**Контекст.** Преамбула в [flat_renderer.py:flat_begin](src/plot_nn_mcp/flat_renderer.py) объявляет:
- `smallblock` style — никто не использует (никакой `flat_block(..., style="smallblock")` нигде)
- `clrbackground` цвет — определяется, не используется в `fill=` нигде
- `clrdense` цвет — равен `clrresidual` hex'ом (см. [themes.py](src/plot_nn_mcp/themes.py)), две переменные = один цвет

**Цель.** Удалить.

**Файлы:**
- `src/plot_nn_mcp/flat_renderer.py` — `flat_begin()` сократить
- `src/plot_nn_mcp/themes.py` — убрать `background`, `dense` поля из `Theme`
- `src/plot_nn_mcp/lowering.py` — заменить `Role.DENSE` на `Role.RESIDUAL` (или `Role.FFN`)
- `tests/test_dsl.py` — возможно, тесты на тему упоминают эти поля

**Подход.**

```python
# В flat_begin() убрать строку:
# r"    smallblock/.style={block, ...},"

# В theme_to_tikz_colors() — пропустить background, dense:
SKIPPED_FIELDS = {"name", "background", "dense"}
for f in dataclasses.fields(theme):
    if f.name in SKIPPED_FIELDS:
        continue
    ...
```

**Гочи:**
1. **Теста сравнения с golden** — после удаления преамбульных строк все 25 golden'ов сломаются. Регенерируйте через `UPDATE_GOLDEN=1`.
2. **Обратная совместимость.** Если внешний код использует `theme.dense` или `theme.background` напрямую — сломается. Сейчас никто не использует, но в Phase 2 я оставил для safety. Можно сделать deprecated через `__getattribute__` warning.
3. `Role.DENSE` → переименовать в `Role.LINEAR` или просто использовать `Role.FFN`.

**Тесты:**
- `tests/test_dsl.py` upd: `test_theme_applied` — убрать assertions про `clrdense`/`clrbackground`.
- Новый тест: `test_no_dead_styles_in_preamble` — `assert "smallblock" not in tex` для всех baselines.

**Критерий приёмки:**
- 25 golden'ов регенерированы.
- `grep -r "clrdense\|clrbackground\|smallblock" tests/golden/` → ничего.
- `pytest tests/` → all green.

**Оценка:** ~1 час.

---

### Задача 9: Property-based tests через Hypothesis

**Контекст.** Существующие тесты — на конкретные архитектуры. Property tests дадут защиту от сценариев, которые мы не подумали проверить.

**Цель.** Hypothesis-based генератор `Architecture` + инварианты.

**Файлы:**
- `tests/test_property.py` (новый)
- `pyproject.toml` или `requirements-dev.txt` — добавить `hypothesis`

**Подход.**

```python
from hypothesis import given, strategies as st
from plot_nn_mcp import dsl
from plot_nn_mcp.lowering import architecture_to_ir
from plot_nn_mcp.render import emit_tikz
from plot_nn_mcp.themes import get_theme

@st.composite
def random_layer(draw):
    kind = draw(st.sampled_from(["embed", "tb", "conv", "head"]))
    if kind == "embed":
        return dsl.Embedding(d_model=draw(st.integers(64, 1024)),
                              label=draw(st.text(min_size=1, max_size=20)))
    if kind == "tb":
        return dsl.TransformerBlock(
            attention=draw(st.sampled_from(["self", "masked", "cross"])),
            norm=draw(st.sampled_from(["pre_ln", "post_ln"])),
            heads=draw(st.integers(1, 32)),
            d_model=draw(st.integers(64, 1024)),
        )
    ...

@st.composite
def random_arch(draw):
    n = draw(st.integers(2, 20))
    arch = dsl.Architecture(name=draw(st.text(min_size=1, max_size=20)))
    for _ in range(n):
        arch.add(draw(random_layer()))
    return arch


@given(random_arch())
def test_random_arch_produces_valid_tex(arch):
    tex = arch.render(use_ir=True)
    assert tex.startswith(r"\documentclass")
    assert tex.rstrip().endswith(r"\end{document}")
    assert tex.count(r"\begin{tikzpicture}") == 1


@given(random_arch())
def test_random_arch_no_fractional_cm(arch):
    import re
    tex = arch.render(use_ir=True)
    assert not re.findall(r"\d\.\d{4,}cm", tex)


@given(random_arch())
def test_random_arch_at_most_one_input_output(arch):
    import re
    tex = arch.render(use_ir=True)
    assert len(re.findall(r"\{Input\}", tex)) <= 1
    assert len(re.findall(r"\{Output\}", tex)) <= 1
```

**Гочи:**
1. Hypothesis может сгенерить `dsl.TransformerBlock` без обязательного label. Проверьте, что dataclass принимает все опции.
2. Random label с `\\` или `$` могут сломать LaTeX escape. Ограничьте text strategy буквами и пробелами.
3. Очень длинные архитектуры могут таймаутить. Ограничьте `max_size=20`.

**Критерий приёмки:**
- `pytest tests/test_property.py -v` → 3+ property tests, каждый с >100 examples без падений.
- `coverage()` не регрессирует.

**Оценка:** ~3 часа (нужно знание Hypothesis).

---

### Задача 10: Multi-type pattern collapse edge cases

**Контекст.** [dsl.py:_detect_groups](src/plot_nn_mcp/dsl.py) детектит и multi-type patterns — например, `[A, B, A, B, A, B]` → `pattern=[A, B] × 3`. Мой `architecture_to_ir` использует `pattern_len`, но не уверен что edge cases работают.

**Цель.** Покрыть тестами и пофиксить.

**Файлы:**
- `tests/test_lowering.py`
- `src/plot_nn_mcp/lowering.py` (если нужны фиксы)

**Подход.**

Прочитать `_detect_groups` внимательно. Тестировать:
- Pattern len=2, count=5, show_n=2 — должно быть 2 копии паттерна (4 layers visible) + ×5 badge.
- Smaller pattern wraps larger pattern (вложенные группы).
- Pattern прерывается structural layer (SectionHeader) — не должен слепляться.

```python
def test_multi_type_pattern_collapses_correctly():
    arch = Architecture("test")
    for i in range(6):
        arch.add(TransformerBlock())
        arch.add(Activation("relu"))
    arch.add(ClassificationHead())
    g = architecture_to_ir(arch, show_n=2)
    # show_n=2 means 2 of [TB, Act] = 4 visible layers
    transformer_count = sum(1 for n in g.nodes.values()
                            if n.label and "Self-Attention" in n.label)
    assert transformer_count == 2
    # Group should have repeat_count=6
    grp = next(iter(g.groups.values()))
    assert grp.repeat_count == 6
```

**Оценка:** ~2-3 часа.

---

### Задача 11: Документация

**Контекст.** Ни README, ни CHANGELOG не обновлены. Пользователь не знает про IR-pipeline.

**Цель.** Минимально:
- README: упоминание трёхслойной архитектуры, как написать новый block-type
- CHANGELOG: записать изменения этого рефактора
- MCP server schema (см. задачу 12)
- Docstring'и в публичных модулях

**Файлы:**
- `README.md`
- `CHANGELOG.md` (создать если нет)
- Обновить docstring в `src/plot_nn_mcp/__init__.py`

**Подход.**

README структура:
```markdown
# plot_nn_mcp

## Architecture

Three layers:
1. DSL — `Architecture(name).add(Layer)`
2. IR — DAG with composable Residual / Parallel / Sequence operators
3. Render — IR → TikZ string

[diagram or ASCII art]

## Adding a new block type

```python
from plot_nn_mcp.lowering import register
from plot_nn_mcp.ir import IRBlockOp, IRResidualOp
from plot_nn_mcp.themes import Role

@dataclass
class MyLayer:
    width: int = 64
    label: str = "MyLayer"

def my_layer_to_ir(layer):
    return IRBlockOp(role=Role.ATTENTION, label=layer.label, dim=layer.width)

register(MyLayer, my_layer_to_ir)
```

## Migration from legacy

Default render path is now `use_ir=True`. To revert temporarily:
```python
arch.render(use_ir=False)  # legacy path (deprecated)
```
```

**Оценка:** ~2-3 часа.

---

### Задача 12: MCP server schema update

**Контекст.** [server.py](src/plot_nn_mcp/server.py) описывает блоки строковыми color-role аргументами для LLM. Стоит обновить enum видимый в schema.

**Файлы:** `src/plot_nn_mcp/server.py`.

**Подход.**

```python
# В Pydantic models server.py:
from plot_nn_mcp.themes import Role

class CustomBlockArgs(BaseModel):
    text: str
    color_role: Role = Role.DENSE  # was: str = "dense"
```

LLM-клиент через MCP introspection увидит валидные значения `Role`.

**Гочи:**
1. Pydantic + StrEnum может потребовать `use_enum_values=True` в Config.
2. JSON schema export должен показывать enum values, не type names.

**Оценка:** ~1-2 часа.

---

## 5. Workflow rules — обязательно соблюдайте

### 5.1 Цикл работы над каждой задачей

```
1. Прочитать relevant test файлы — понять что уже покрыто
2. Написать новый failing test (red)
3. Минимальная имплементация чтобы тест прошёл (green)
4. Запустить ВСЕ тесты — `pytest tests/ -q` должно быть 590+ passed
5. Если golden-тесты падают:
   a) Если изменение НАМЕРЕННОЕ → `UPDATE_GOLDEN=1 pytest tests/test_golden.py::test_snapshot_matches_golden -q`
   b) Затем `git diff tests/golden/` — посмотреть глазами все ли изменения ожидаемы
   c) Если есть нежеланные изменения — править код, не golden
6. Refactor (если есть дубли)
7. Финал: pytest tests/ -q → all green → commit
```

### 5.2 Что НЕ делать

- ❌ **Не меняйте публичный API** `Architecture.add().render()`. Все 47 layer dataclass'ов остаются.
- ❌ **Не удаляйте `LEGACY_ONLY` пока не мигрируете тип.** Этот frozenset — единственная защита от молчаливого fall back на legacy.
- ❌ **Не запускайте `UPDATE_GOLDEN=1` без последующего `git diff`.** Иначе вы пропустите визуальные регрессии.
- ❌ **Не удаляйте тесты без обоснования в коммит-сообщении.** Каждый тест ловит конкретный баг или регрессию.
- ❌ **Не используйте `# type: ignore`** для Role/IROp типизации. Если mypy ругается — значит реально что-то не так.

### 5.3 Что ВСЕГДА делать

- ✅ Перед коммитом: `pytest tests/ -q`
- ✅ После каждого изменения lowering — обновить `coverage()` порог в `test_coverage_progress`.
- ✅ Новый layer type → register в `_LOWERING` ИЛИ добавить в `LEGACY_ONLY` — иначе `coverage["migrated"] + ["legacy_only"] != ["total"]` и тест падает.
- ✅ Новый IR-инвариант → добавить тест в `tests/test_ir.py`.
- ✅ Структурные изменения emit_tikz → проверить ВСЕ 25 baseline на дробные cm и double Output.

---

## 6. Pitfalls и gotchas — то, что прихватил при рефакторе

### 6.1 Почему `output_node` пиннится **до** badge anchors

В `architecture_to_ir`:
```python
if builder.cursor is not None:
    builder.set_output(builder.cursor)  # <-- ПЕРЕД badges
for grp in groups:
    if grp.count > show_n:
        ...add badge anchor...
```

Если pin сделать **после** добавления badge'ей, `IRBuilder.build()` поставит `output_node = order[-1]` которое будет badge anchor (роль RESIDUAL, label "$\times12$"). Тогда `_redundant("Output")` вернёт False, и default Output arrow добавит лишнюю метку.

Не переставляйте этот pin без понимания.

### 6.2 Почему `Embedding(rope=True)` в lowering автоматически добавляет `RoPE` ⊕

Legacy [dsl.py:2179-2191](src/plot_nn_mcp/dsl.py) сам добавляет RoPE circle если `Embedding.rope=True` И следующий layer не PositionalEncoding. Тест `test_modernbert_pattern` в [test_dsl.py](tests/test_dsl.py) полагается на это. Поэтому в `embedding_to_ir`:
```python
if not layer.rope:
    return block
return IRSequenceOp(ops=[
    block,
    IRBlockOp(role=Role.RESIDUAL, label="RoPE", shape="circle"),
])
```

**Не упрощайте до просто IRBlockOp** — сломаете тест.

### 6.3 BERT использует `post_ln`, не `pre_ln`

Когда я писал lowering, было соблазн думать «BERT = pre_ln». Но в [generate_all.py:97](examples/generate_all.py#L97) BERT использует `TransformerBlock("self", "post_ln", ...)`. Поэтому:
```python
def transformer_block_to_ir(block):
    is_pre = block.norm == "pre_ln"  # check this carefully
    if is_pre:
        # LN → Attn → +
    else:
        # Attn → + → LN  (this is BERT/RoBERTa/DeBERTa)
```

### 6.4 `flat_block(fill="attention!85")` — pre-composed opacity

В legacy некоторые callsites передают `fill="attention!85"` напрямую (с уже включённой opacity). `resolve_fill` детектит наличие `!` и пропускает:
```python
if "!" in fill:
    return fill  # leave as-is
```

При миграции на `Role` enum нельзя передавать `Role.ATTENTION!85` — это синтаксическая ошибка. Опцию opacity передавайте через параметр `opacity=0.85` в `flat_block`.

### 6.5 Width rounding только в emit, не в lowering

Я делал `round(w, 2)` в `_emit_node` (см. [render.py](src/plot_nn_mcp/render.py)). Не округляйте в `lowering.py` — потеряете точность для будущих layout-стратегий, которые могут использовать `size_hint` для расчётов перед эмиссией.

### 6.6 `_detect_groups` пересекает structural layers

Структурные слои (`SectionHeader`, `Separator`) прерывают группировку — это намеренно. Если будете расширять `_detect_groups`, не убирайте эту проверку: иначе SectionHeader попадёт внутрь pattern'а и сломается layout.

### 6.7 Test isolation: `PLOTNN_SKIP_WRITE`

[examples/generate_all.py](examples/generate_all.py) при импорте создаёт файлы. Тест `test_count_baseline_architectures_lowerable` устанавливает `os.environ["PLOTNN_SKIP_WRITE"] = "1"` ДО импорта, чтобы избежать. Если вы делаете аналогичный тест — обязательно дублируйте этот пре-импорт setup.

### 6.8 `coverage()` инвариант ловит «забытые» layer types

Если добавите в [dsl.py](src/plot_nn_mcp/dsl.py) новый dataclass, но не зарегистрируете его ни в `_LOWERING`, ни в `LEGACY_ONLY` — `test_coverage_progress` упадёт с ошибкой типа `migrated + legacy != total`. Это feature, не баг. Решение: добавить тип в один из двух наборов.

### 6.9 Goldens ≠ ожидаемое поведение

Текущие goldens — это **фиксация текущего состояния**, не «правильный» вывод. Они нужны чтобы поймать **непреднамеренные** изменения. Если вы намеренно меняете output (фикс бага, новая фича) — обновите goldens и ОБЯЗАТЕЛЬНО посмотрите git diff чтобы убедиться что изменения именно те.

### 6.10 IR Edge ordering

`graph.edges` хранится списком в порядке добавления, не сгруппирован по типу. При сериализации в emit_tikz я перебираю их линейно. Если планируете layout-pass который пересобирает edges (например, упорядочивает skip-edges по target node для меньшей перекрёстности arrows) — сортировка должна быть **стабильной** относительно текущего порядка, иначе golden может flake.

---

## 7. Verification strategy — как убедиться что задача сделана

Для каждой задачи:

1. **Unit-тест в `tests/test_<module>.py`** — добавляется обязательно
2. **Integration-тест** — где возможно (`test_<feature>_smoke` в [tests/test_render.py](tests/test_render.py) или [tests/test_ir_path_fixes.py](tests/test_ir_path_fixes.py))
3. **Полный pytest run** — `pytest tests/ -q` → 590+ зелёные
4. **Golden update + diff** — если output меняется
5. **Coverage assertion** — поднимите `test_coverage_progress` порог если мигрировали тип
6. **Manual file inspection** — посмотрите регенерированный baseline `examples/output/architectures/<арх>.tex` — структура выглядит разумно
7. **Optional: PDF compilation** — если LaTeX установлен, скомпилируйте регенерированные `.tex` через `pdflatex`/`tectonic` и визуально проверьте

### Финальная sanity check после ВСЕХ задач

```bash
# Все тесты
pytest tests/ -q

# Полная регенерация baselines
rm -rf examples/output/architectures
python examples/generate_all.py

# Проверки выходов
cd examples/output/architectures
echo "=== fractional cm ==="
ls *.tex | xargs grep -lE "[0-9]\.[0-9]{4,}cm" | wc -l  # должно быть 0

echo "=== double Output ==="
for f in *.tex; do c=$(grep -c "{Output}" "$f"); [ "$c" -gt 1 ] && echo "$f: $c"; done  # пусто

echo "=== Mixtral residuals ==="
grep -c skiparrow 11_mixtral_8x7b.tex  # > 0

echo "=== group frames ==="
grep -c "fit=" *.tex | awk -F: '{s+=$2}END{print s}'  # ожидаемо ~44

echo "=== ×N badges ==="
grep -c "_badge" *.tex | awk -F: '{s+=$2}END{print s}'  # ~23

# Count line reduction
wc -l ../../src/plot_nn_mcp/dsl.py  # ожидаемо < 1500 после задачи 7
```

---

## 8. Priority и оценки времени

| # | Задача | Сложность | Время | Pre-req | Эффект |
|---|---|---|---|---|---|
| 1 | LSTM/GRU/Mamba decompose | средняя | 6h | — | -3 baseline regression |
| 2 | Parallel layout in emitter | сложная | 5h | — | unblocks #5/#6 visual |
| 3 | Horizontal/UNet via IR | сложная | 6h | #6 partial | покрывает все layouts |
| 4 | Skip xshift from bbox | маленькая | 2h | — | визуальная аккуратность |
| 5 | EncoderDecoder/SideBySide etc | средняя | 8h | #2 | расширяет coverage |
| 6 | UNetLevel/Bottleneck etc | средняя | 8h | #2, #3 | финальные типы |
| 7 | Удалить legacy `_render_*` | маленькая | 3h | #1, #5, #6 | -1500 строк |
| 8 | Cleanup dead preamble | тривиальная | 1h | — | косметика |
| 9 | Hypothesis property tests | средняя | 3h | — | долгосрочная защита |
| 10 | Multi-type pattern edges | средняя | 3h | — | edge-case coverage |
| 11 | Documentation | средняя | 3h | — | onboarding |
| 12 | MCP server schema | маленькая | 2h | — | LLM UX |

**Рекомендуемый порядок:** 1 → 4 → 8 → 2 → 5 → 6 → 7 → 11 → 10 → 9 → 12 → 3.

Обоснование:
- **1, 4, 8** — короткие выигрыши без зависимостей (3 baseline fix, layout cleanup, dead code).
- **2** — разблокирует красивую отрисовку для #5/#6.
- **5, 6** — после #2 идут естественно (нужен параллельный layout).
- **7** — финальная чистка после полного coverage.
- **11** — после стабилизации архитектуры документировать.
- **10** — низкоприоритетные edge cases.
- **9, 12** — отдельные направления, можно делать в любое время.
- **3** — самая большая (UNet U-shape), оставьте на потом, когда есть время погрузиться.

---

## 9. Финальный sanity-список перед закрытием рефактора

После всех 12 задач должно быть:

- [ ] `wc -l src/plot_nn_mcp/dsl.py` < 1500
- [ ] `coverage()` показывает `migrated == total == 47`, `legacy_only == 0`
- [ ] `pytest tests/` → 700+ passed (новые тесты добавлены), 0 failed
- [ ] Все 25 baseline `.tex` без `\d\.\d{4,}cm`
- [ ] Все 25 baseline с ровно 1 `{Input}` и ≤1 `{Output}`
- [ ] Каждая `IRResidualOp` body доказана тестом-инвариантом
- [ ] README обновлён
- [ ] CHANGELOG fixed
- [ ] Параметр `use_ir` deprecated/удалён из `Architecture.render`
- [ ] Hypothesis property tests на 3+ инварианта проходят 100+ examples
- [ ] MCP server schema показывает Role enum

---

## 10. Контактная информация о структуре кода

### Файлы и их назначение

| Файл | Назначение | Размер |
|---|---|---|
| [src/plot_nn_mcp/dsl.py](src/plot_nn_mcp/dsl.py) | DSL dataclass'ы + legacy renderer | ~2870 строк (после задачи 7: ~1400) |
| [src/plot_nn_mcp/ir.py](src/plot_nn_mcp/ir.py) | IR types + builder | ~280 строк |
| [src/plot_nn_mcp/render.py](src/plot_nn_mcp/render.py) | IR → TikZ emitter | ~190 строк |
| [src/plot_nn_mcp/lowering.py](src/plot_nn_mcp/lowering.py) | DSL → IR dispatch | ~390 строк |
| [src/plot_nn_mcp/themes.py](src/plot_nn_mcp/themes.py) | Themes + Role enum | ~210 строк |
| [src/plot_nn_mcp/flat_renderer.py](src/plot_nn_mcp/flat_renderer.py) | TikZ primitives (flat_block, flat_arrow, ...) | ~460 строк |
| [src/plot_nn_mcp/server.py](src/plot_nn_mcp/server.py) | MCP server | ~400 строк |
| [src/plot_nn_mcp/presets.py](src/plot_nn_mcp/presets.py) | Pre-built architecture factories | ~870 строк |
| [src/plot_nn_mcp/compiler.py](src/plot_nn_mcp/compiler.py) | Compiles .tex → .pdf | ~90 строк |
| [tests/test_golden.py](tests/test_golden.py) | Snapshot tests + invariants | ~140 строк |
| [tests/test_ir.py](tests/test_ir.py) | IR-level tests | ~120 строк |
| [tests/test_render.py](tests/test_render.py) | IR → TikZ tests | ~80 строк |
| [tests/test_lowering.py](tests/test_lowering.py) | Lowering tests | ~190 строк |
| [tests/test_ir_path_fixes.py](tests/test_ir_path_fixes.py) | Bug-fix proofs (legacy vs IR) | ~180 строк |
| [tests/test_dsl.py](tests/test_dsl.py) | DSL public API tests | ~450 строк |
| [tests/test_e2e.py](tests/test_e2e.py) | End-to-end through full pipeline | ~900 строк |
| [tests/test_new_architectures.py](tests/test_new_architectures.py) | New-architecture coverage | ~630 строк |
| [tests/test_server.py](tests/test_server.py) | MCP server tests | ~220 строк |
| [tests/golden/](tests/golden/) | 25 .tex snapshots | — |
| [examples/generate_all.py](examples/generate_all.py) | 25 baseline architectures | ~440 строк |

### Ключевые публичные интерфейсы

- `dsl.Architecture(name, theme, layout)` + `.add(layer)` + `.render(show_n, use_ir)` + `.render_to_file(...)`
- `ir.IRBuilder(title)` + `.add_block(...)` + `.add_op(IROp)` + `.set_input(id)` + `.set_output(id)` + `.build()`
- `lowering.layer_to_ir(layer) → IROp`
- `lowering.architecture_to_ir(arch, show_n) → IRGraph`
- `lowering.can_lower_architecture(arch) → (bool, [missing_types])`
- `lowering.register(LayerType, fn)` — для пользовательских lowering
- `lowering.coverage() → {migrated, legacy_only, total}`
- `render.emit_tikz(graph, theme) → str`
- `themes.Role` (enum), `themes.resolve_fill(role_or_str) → str`

---

## 11. Если что-то идёт не так

### "590 тестов было, а сейчас 580"
Кто-то удалил тесты. Делаем `git log -p tests/` — найдите кто и зачем. Если нет обоснования в commit message — restore.

### "test_coverage_progress падает с 'migrated + legacy != total'"
Добавили layer dataclass, не зарегистрировали. Решение: добавить в `_LOWERING` (через register) или в `LEGACY_ONLY` set.

### "test_no_fractional_cm падает"
Где-то size_hint передаётся напрямую в `\node[minimum width={...}cm]` без округления. Найдите место, добавьте `round(w, 2)`.

### "Mixtral вдруг показывает только 1 skiparrow"
Кто-то изменил `moe_layer_to_ir` чтобы он не использовал `IRResidualOp`. Восстановите. Это критичный structural fix.

### "LaTeX компиляция падает с undefined color"
Theme не определяет цвет, который lowering запросил. Проверьте `themes.py` `Theme` dataclass — все ли роли покрыты. И `theme_to_tikz_colors` эмитит ли все нужные `\definecolor`.

### "Goldens отличаются после `git pull`"
Кто-то забыл закоммитить обновлённые goldens. Сделайте `UPDATE_GOLDEN=1 pytest tests/test_golden.py::test_snapshot_matches_golden`, посмотрите diff, решите — это легитимные изменения или regression.

### "pytest зависает"
Возможно, `examples/generate_all.py` пытается компилировать PDF через `pdflatex` (если он установлен в PATH). Установите `os.environ["PLOTNN_SKIP_WRITE"] = "1"` ДО импорта.

---

## 12. Одна строка резюме

Рефактор `plot_nn_mcp` финализирован до состояния «работает, default IR, все critical bugs закрыты структурно, 590 тестов зелёные, ~12 задач до полной чистоты — следующий агент берёт #1 (LSTM gates), идёт по списку».

Удачи.
