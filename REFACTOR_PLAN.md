# Deep Refactor Plan — plot_nn_mcp

## 0. Контекст и цель

Текущее устройство ([src/plot_nn_mcp/dsl.py](src/plot_nn_mcp/dsl.py), 2867 строк; [src/plot_nn_mcp/flat_renderer.py](src/plot_nn_mcp/flat_renderer.py), 456 строк) — однопроходный транслятор `list[Layer] → str TikZ`. Центр — цикл с 50+ ветками `isinstance(layer, X)`, разделяющими переменную `prev: str` и напрямую конкатенирующими TikZ-строки. Такой стиль породил класс концептуальных ошибок:

- **Граф, layout и стиль смешаны** в одном вызове рендера блока.
- **Residual / Norm / Section не являются first-class примитивами** — они зашиты внутрь рендереров отдельных блоков. Следствие: Mixtral MoE без add2, DeBERTa с двойным «Output», YOLOv8 с последовательными detect-головами вместо параллельных.
- **`first_node = prev`** не различает семантические и служебные узлы → Input-стрелка цепляется к SectionHeader.
- **Цветовые слоты используются позиционно** (`"attention"` значит то self-attention, то Conv, то GraphConv) → ломается язык диаграммы.
- **Дробные сантиметры** текут из `width_from_dim` прямо в TikZ (нет слоя computed layout).

Цель рефакторинга — ввести три чётко разделённых слоя: **IR → Layout → Render**, сделать композиционные примитивы `Residual` / `Group` / `Section` first-class, унифицировать цвет через `role: Role`. Все текущие публичные API (`Architecture`, `.add`, `.render`, MCP server, presets) должны продолжать работать — рефакторинг внутренний.

Рефактор разбит на **7 фаз**, каждая из которых мержабельна независимо и сохраняет зелёными тесты [tests/test_dsl.py](tests/test_dsl.py), [tests/test_e2e.py](tests/test_e2e.py), [tests/test_new_architectures.py](tests/test_new_architectures.py), [tests/test_server.py](tests/test_server.py). После каждой фазы перегенерируются 25 архитектур [examples/generate_all.py](examples/generate_all.py) и сравниваются с baseline (хотя бы на уровне "все 25 .tex валидны, количество nodes в рамках ±5%").

---

## Фаза 1. Golden baseline + test safety net (1 коммит)

**Проблема, которую решаем до рефакторинга:** любое движение без baseline-снапшотов вслепую ломает существующие диаграммы, и мы этого не заметим — LaTeX на dev-машине отсутствует.

### Шаги

1. Создать `tests/golden/` со снапшотами .tex для всех 25 архитектур (скопировать текущие [examples/output/architectures/*.tex](examples/output/architectures/)).
2. Добавить `tests/test_golden.py`:
   - Для каждой архитектуры из `generate_all.py`: рендер → сравнить с golden → diff при расхождении.
   - Флаг `UPDATE_GOLDEN=1` пересоздаёт снапшоты.
3. Починить тихое падение компиляции в [generate_all.py:19-24](examples/generate_all.py#L19-L24): при отсутствии `pdflatex`/`tectonic` печатать один раз `"(no LaTeX engine found — skipping PDF)"`.
4. Добавить структурные инварианты в `test_golden.py`: для каждого .tex проверять
   - `\begin{document}` / `\end{document}` сбалансированы;
   - количество `\node[block,...]` равно ожидаемому (детерминистично от модели);
   - отсутствие дробных `cm` с 4+ знаками;
   - каждая `skiparrow` имеет соответствующий `add_circle`;
   - `Input`/`Output` метки присутствуют ровно по одной.

**Деливери:** после этой фазы любой последующий коммит, сломавший визуал, падает на CI. Без этого остальные фазы опасны.

**Файлы:** `tests/golden/*.tex` (новое), `tests/test_golden.py` (новое), [examples/generate_all.py](examples/generate_all.py).

---

## Фаза 2. Цветовая семантика — `Role` enum (1 коммит, чисто типовой)

**Концептуальная проблема #3 (цвет):** каждая ветка `isinstance` сама выбирает цветовой slot. Чинится до структурного рефакторинга — это изолированная операция.

### Шаги

1. В [themes.py](src/plot_nn_mcp/themes.py) добавить:
   ```python
   class Role(StrEnum):
       INPUT = "input"           # Embedding, Noise, Image
       ATTENTION = "attention"   # self-/cross-/masked-attention ONLY
       CONV = "conv"             # ConvBlock, CSPDarknet, Swin window-attn as conv-like
       RECURRENT = "recurrent"   # LSTM, GRU, Mamba
       FFN = "ffn"               # FFN, MLP, Dense, Experts
       NORM = "norm"             # LayerNorm, RMSNorm, BatchNorm
       POOL = "pool"             # MaxPool, AvgPool, PatchMerging
       RESIDUAL = "residual"     # only for add/skip symbols
       GATE = "gate"             # Router, Gate, Attention weights
       SPECTRAL = "spectral"     # Fourier, wavelet
       PHYSICS = "physics"       # PINN, DeepONet trunk
       OUTPUT = "output"         # ClassificationHead, LM Head
       GROUP = "group"           # section/frame background
   ```
   `Theme` становится `dict[Role, str]` (hex). Старые поля (`attention`, `ffn`, ...) убираются — тесты темы ([tests/test_dsl.py](tests/test_dsl.py)) обновляются.
2. Добавить функцию `resolve_fill(role: Role, theme: Theme) -> str` — единственный путь получения цвета.
3. Убрать из [flat_renderer.py:119](src/plot_nn_mcp/flat_renderer.py#L119) строковый параметр `fill: str`; заменить на `role: Role`. Все вызовы `flat_block(... "attention")` проходят через `resolve_fill`.
4. Слить `dense` и `residual` (один цвет → одна роль `RESIDUAL`, `DenseLayer` использует `Role.FFN`). Убрать `clrdense` из преамбулы.
5. Убрать мёртвый `smallblock` стиль из `flat_begin()` (никто не использует).
6. `clrbackground` используется? Если нет — убрать из `theme_to_tikz_colors`.

**Что ломает:** ничего визуально — цвета остаются те же, но теперь привязаны к семантике, а не к позиции в списке аргументов. Golden-снапшоты должны остаться побайтно идентичными (кроме преамбулы — убираем dead code).

**Деливери:** `grep "attention" *.tex` больше не означает Conv или GraphConv. Один цвет = одна роль.

**Файлы:** [themes.py](src/plot_nn_mcp/themes.py), [flat_renderer.py](src/plot_nn_mcp/flat_renderer.py), все 50+ `isinstance` веток в [dsl.py](src/plot_nn_mcp/dsl.py) (поиск-замена строковых литералов на `Role.X`).

---

## Фаза 3. Ввести IR (Intermediate Representation)

**Концептуальная проблема #1 (граф):** `prev: str` — это список, а не DAG. Нет понятия точек ветвления и параллельных веток.

### Модель IR

Создаём новый модуль [src/plot_nn_mcp/ir.py](src/plot_nn_mcp/ir.py):

```python
@dataclass
class IRNode:
    id: str                        # стабильный, генерится детерминистично
    role: Role                     # для цвета + семантики
    label: str
    shape: ShapeHint               # "block" | "circle" | "anchor" | "section_rule"
    size_hint: SizeHint            # (width, height) или ("auto", "auto")
    dim: int | None = None         # для боковой метки-размерности
    meta: dict = field(default_factory=dict)  # layer-specific (e.g., kernel_size)

@dataclass
class IREdge:
    src: str
    dst: str
    kind: Literal["data", "skip", "branch", "merge"]
    style: dict = field(default_factory=dict)   # color override, label, rounded

@dataclass
class IRGroup:
    id: str
    children: list[str]            # node ids
    title: str | None = None       # если задан — отображается как SectionHeader
    repeat_count: int | None = None   # для ×N
    subgroups: list[str] = field(default_factory=list)  # вложенные группы

@dataclass
class IRGraph:
    nodes: dict[str, IRNode]
    edges: list[IREdge]
    groups: dict[str, IRGroup]
    input_node: str | None         # явно, не угадывается
    output_node: str | None        # явно
    order: list[str]               # топологический/вертикальный порядок для линейных кусков
```

Ключевые отличия от сегодняшнего состояния:

- `input_node` и `output_node` **устанавливаются явно** при построении, а не восстанавливаются из `first_node = prev`. Ломаем регрессию коммита `d6fda3e`.
- Группы могут иметь `title` → SectionHeader становится *свойством* группы, а не отдельным слоем. Фикс DCGAN (секции без фрейма).
- Рёбра типизированы: `data` (обычная стрелка), `skip` (skiparrow), `branch`/`merge` (параллельные ветки для YOLOv8).

### Шаги

1. Написать `ir.py` с dataclass'ами выше, `IRBuilder` helper-классом с fluent API:
   ```python
   b = IRBuilder(theme)
   emb = b.add_block(role=Role.INPUT, label="Token Embedding", dim=768)
   b.set_input(emb)
   res_block = b.residual(body=lambda inner:
       inner.add_block(role=Role.NORM, label="LayerNorm")
            .add_block(role=Role.ATTENTION, label="Self-Attn"))
   b.connect(emb, res_block.entry)
   ```
2. Добавить генератор IR → TikZ: новый файл [src/plot_nn_mcp/render.py](src/plot_nn_mcp/render.py). Пока — простая вертикальная layout-стратегия, точно воспроизводящая текущий вид.
3. Адаптер совместимости: старый `_render_vertical(layers, ...)` внутри превращает `list[Layer]` в `IRGraph` через новый `layers_to_ir()`, затем вызывает `render.py`. Публичный API `Architecture.render()` не меняется.

**Деливери:** тот же TikZ-выхлоп (golden-тесты зелёные), но теперь есть промежуточный слой, на котором можно оперировать графом. Никто снаружи модуля не видит.

**Файлы:** новые — `ir.py`, `render.py`; изменяется — [dsl.py](src/plot_nn_mcp/dsl.py) (функции `_render_vertical`, `_render_horizontal`).

---

## Фаза 4. First-class `Residual`, `Norm`, `Parallel` — композируемые примитивы

**Концептуальная проблема #2 (композиция):** 50+ монолитных `isinstance` веток, residual зашит внутрь `_render_transformer_block` и `_render_moe`. Чиним корень проблемы Mixtral/MoE.

### Новые IR-операторы (в [ir.py](src/plot_nn_mcp/ir.py))

```python
@dataclass
class IRResidualOp:
    """Оборачивает поддерево в residual: entry → body → add ← skip(entry, add)."""
    body: list[IROp]

@dataclass
class IRSequenceOp:
    """Последовательность (default composition)."""
    ops: list[IROp]

@dataclass
class IRParallelOp:
    """Split → N веток → optional merge. Для YOLOv8 detect-heads."""
    branches: list[IROp]
    merge: Literal["concat", "add", "none"] = "none"

@dataclass
class IRBlockOp:
    """Лист — IRNode, создаваемый при lowering."""
    role: Role; label: str; size_hint: SizeHint; ...
```

`IROp = IRResidualOp | IRSequenceOp | IRParallelOp | IRBlockOp`. Есть шаг **lowering**: `IROp` → `IRGraph` (узлы + рёбра). `IRResidualOp` генерирует: `entry_anchor`, `body_nodes`, `add_circle`, edge `(body_last → add, data)`, edge `(entry → add, skip)`.

### Переписанные ветки

1. **`TransformerBlock`** ([dsl.py:656-762 — 107 строк](src/plot_nn_mcp/dsl.py#L656)) → ~15 строк:
   ```python
   def transformer_block_to_ir(block: TransformerBlock) -> IROp:
       attn = IRResidualOp([
           IRBlockOp(Role.NORM, "LayerNorm") if block.norm=="pre_ln" else None,
           IRBlockOp(Role.ATTENTION, attn_label(block)),
       ])
       if block.skip_ffn:
           return IRSequenceOp([attn])
       ffn = IRResidualOp([
           IRBlockOp(Role.NORM, "LayerNorm") if block.norm=="pre_ln" else None,
           IRBlockOp(Role.FFN, ffn_label(block)),
       ])
       post = [IRBlockOp(Role.NORM, "LayerNorm")] if block.norm=="post_ln" else []
       return IRSequenceOp([attn, ffn, *post])
   ```
2. **`MoELayer`** ([dsl.py:2605-2617](src/plot_nn_mcp/dsl.py#L2605-L2617)) → переписать как residual (фикс Mixtral):
   ```python
   def moe_to_ir(layer: MoELayer) -> IROp:
       return IRResidualOp([
           IRBlockOp(Role.NORM, "LayerNorm"),
           IRBlockOp(Role.GATE, f"Router (top-{layer.top_k})"),
           IRBlockOp(Role.FFN, f"{layer.num_experts} Experts (FFN {layer.d_ff})"),
       ])
   ```
   Теперь в Mixtral добавится `add2` и `ln2` автоматически — residual в одном месте, работает для attn и для MoE.
3. **`ResidualBlock`**, **`BottleneckBlock`** ([dsl.py:2396,2420](src/plot_nn_mcp/dsl.py#L2396)) → тоже через `IRResidualOp`.
4. **`LSTMBlock`/`GRUBlock`/`MambaBlock`** — оставить монолитными рендерерами (они рисуют гейты), но возвращать `IROp` через специальный `IRCustomOp` — чтобы не выпадать из pipeline.

### Параллельные ветки

5. **YOLOv8 detect heads** — заменить 3 последовательных блока на `IRParallelOp([P3, P4, P5], merge="concat_output")`. Layout-стратегия разложит их горизонтально.
6. **SideBySide**, **BidirectionalFlow** ([dsl.py:1232,1251](src/plot_nn_mcp/dsl.py#L1232)) — становятся тонкими обёртками над `IRParallelOp`.
7. **Generator/Discriminator** DCGAN — становятся двумя последовательностями внутри `IRGroup` с `title` → SectionHeader рисуется как заголовок группы, а не отдельный слой.

**Деливери:**
- `grep -c 'isinstance(layer,' dsl.py`: с ~55 до ~35 (выигрыш от того, что Residual/Norm общие).
- Mixtral получает add2 автоматически (golden-снапшот обновляется — это первое намеренное визуальное изменение).
- `_render_transformer_block` удаляется целиком.

**Файлы:** [ir.py](src/plot_nn_mcp/ir.py), [render.py](src/plot_nn_mcp/render.py), [dsl.py](src/plot_nn_mcp/dsl.py) (большая часть `isinstance` веток).

---

## Фаза 5. Layout-слой — отдельно от рендера

**Концептуальная проблема #3 (layout):** `width_from_dim` возвращает float прямо в TikZ (дробные cm). Skip-offsets 2.3/2.45/2.6 — магические числа. Стрелки могут вылезать за group frame. Нет глобального взгляда на позиции.

### Модель

Новый модуль [src/plot_nn_mcp/layout.py](src/plot_nn_mcp/layout.py):

```python
@dataclass
class Box:
    x: float; y: float        # центр
    w: float; h: float        # размеры (всегда округляются до 2 знаков при emission)

@dataclass
class Layout:
    boxes: dict[str, Box]     # node_id → Box
    routes: dict[int, list[tuple[float, float]]]   # edge_idx → polyline
    groups: dict[str, Box]    # group_id → bounding box (включая padding)

def compute_layout(graph: IRGraph, strategy: LayoutStrategy) -> Layout: ...
```

### Стратегии layout

- `VerticalStackLayout` — текущее поведение (по умолчанию).
- `ParallelLayout` — для `IRParallelOp`: размещает ветки бок о бок, выравнивает высоты.
- `UNetLayout` — существующий `_render_unet` ([dsl.py:1372](src/plot_nn_mcp/dsl.py#L1372)) переписывается как LayoutStrategy.

### Фиксы на уровне layout, а не отдельных багов

1. **Skip-offsets считаются по bounding box группы**, а не магически: `skip_xshift = group_width/2 + safety_margin`. Устраняет риск пересечения рамки группы skiparrow'ом.
2. **Ширины блоков округляются** в `emit_tikz(layout)`: `f"{round(box.w, 2)}cm"`. Устраняет `1.8666666666666667cm`.
3. **`width_from_dim`** перемещается в `layout.py` и становится hint'ом, а не прямым значением: layout-стратегия может решить нормализовать ширины в пределах одной секции (все Conv в одной группе → одинаковая ширина, даже если filters разные). Устраняет "Stem Conv 1.87cm рядом с Stage1 3.8cm".
4. **Input/Output стрелки** рисуются layout'ом на основе `graph.input_node` и `graph.output_node` (явные поля из Фазы 3), а не `first_node = prev`. Устраняет Input → SectionHeader.
5. **`SectionHeader` рисуется как заголовок `IRGroup`** — одна функция, один стиль. Устраняет несогласованность стилей разделителей (0.4pt vs 1.2pt в DCGAN).

**Деливери:**
- Нет дробных cm в выхлопе.
- DCGAN: Generator и Discriminator — две группы с заголовками и рамками (golden обновляется).
- YOLOv8 detect-головы визуально параллельны.
- Все косметические баги из предыдущего анализа уходят автоматически — на уровне layout.

**Файлы:** новый [layout.py](src/plot_nn_mcp/layout.py), переписанный [render.py](src/plot_nn_mcp/render.py) (теперь только emit TikZ из Layout).

---

## Фаза 6. Group unification, SectionHeader убираем как отдельный layer

**Концептуальная проблема #2 (продолжение):** `SectionHeader` и `pattern_group` — два несовместимых механизма. [_detect_groups](src/plot_nn_mcp/dsl.py#L547) ищет повторяющиеся паттерны, `SectionHeader` — явный маркер пользователя. Они не знают друг о друге.

### Шаги

1. В [dsl.py](src/plot_nn_mcp/dsl.py) `SectionHeader` остаётся в публичном API **только как алиас** — его `add(SectionHeader("Encoder"))` теперь модифицирует *следующую группу*, а не добавляет слой в общий список. На уровне IR он исчезает.
2. Добавить альтернативный, более явный публичный API:
   ```python
   arch = Architecture("DCGAN")
   with arch.section("Generator"):
       arch.add(...)
       arch.add(...)
   with arch.section("Discriminator"):
       ...
   ```
   Старый `add(SectionHeader(...))` продолжает работать → фиксирует начало секции до следующего SectionHeader или конца.
3. `_detect_groups` работает поверх IR: группировка по `signature` возможна **внутри секции**, но не пересекает секционные границы.
4. `IRGroup.title` отвечает одновременно за:
   - рамку `fit=(...)` (если `len(children) > 1`);
   - заголовок-линию + текст (если `title is not None`);
   - `×N` бейдж (если `repeat_count is not None`).
   Один стиль, одна функция рендера.

**Деливери:** DCGAN и YOLOv8 получают заголовки секций + рамки одновременно. Один стиль разделителей во всём корпусе.

**Файлы:** [dsl.py](src/plot_nn_mcp/dsl.py) (Architecture.section context manager, _detect_groups адаптируется), [render.py](src/plot_nn_mcp/render.py) (одна функция вместо двух для заголовков).

---

## Фаза 7. Cleanup и гарантии

1. **Удалить мёртвый код**: `clrdense` (если слит), `smallblock`, неиспользуемый `clrbackground`, старые монолитные рендереры (`_render_transformer_block`, старый `_render_moe`).
2. **Линтер-инварианты** в `test_golden.py`:
   - Ни в одном `.tex` нет подстроки `[0-9]\.[0-9]{4,}cm`;
   - Каждый `skiparrow` имеет соответствующий `add_circle` (нельзя иметь skip без точки слияния);
   - В каждом файле ровно один `{Input}` и один `{Output}`;
   - Если есть SectionHeader-группа, у неё есть `fit=(...)` рамка.
3. **Обновить MCP tool-схемы** в [server.py](src/plot_nn_mcp/server.py): описания slot'ов (`role`) теперь из `Role` enum.
4. **Обновить 25 golden-снапшотов** — они намеренно меняются (фиксы из Фазы 4/5).
5. **Обновить README/CHANGELOG** с описанием нового публичного API `arch.section(...)`.

---

## Карта соответствия "баг → фаза, которая его чинит"

| Баг (из предыдущего анализа) | Фаза | Механизм фикса |
|---|---|---|
| Input-стрелка → SectionHeader (DCGAN, YOLOv8) | 3 | `IRGraph.input_node` явный |
| Mixtral MoE без add2 | 4 | `IRResidualOp` обёртка |
| YOLOv8 последовательные heads | 4 | `IRParallelOp` |
| DeBERTa дубль Output | 3+5 | `output_node` ставится один раз в IR, default IO-стрелка знает о нём |
| Дробные cm (ResNet/YOLO) | 5 | `round(w, 2)` в emit |
| Skip-arrow вылезает за фрейм группы | 5 | `skip_xshift` из bounding box |
| Stem Conv 1.87cm рядом с Stage 3.8cm | 5 | layout-стратегия нормализует ширины в секции |
| DCGAN без group-frame | 6 | Section = Group |
| Разные стили section rule (0.4pt vs 1.2pt) | 6 | одна функция заголовков |
| `clrdense==clrresidual` | 2 | `Role` enum схлопывает дубли |
| `smallblock` мёртв | 2+7 | убрать из преамбулы |
| Темы плавают (text/background) | 2+7 | `Role` + валидация темы коллекции |
| `clrattention` означает 3 разные сущности | 2 | Conv → `Role.CONV`, Graph → `Role.ATTENTION` и т.д. |
| Тихий пропуск компиляции PDF | 1 | print в `generate_all.py` |

14 из 14 концептуальных и косметических багов из предыдущего анализа попадают в одну из фаз.

---

## Риски и стратегия

1. **Риск регрессии визуала.** Смягчается golden-тестами Фазы 1. Каждая последующая фаза идёт под защитой этих снапшотов.
2. **Риск сломать MCP API.** Публичный API `Architecture/add/render` не трогается; внутренние `_render_*` — да, но они не документированы. Тесты [test_server.py](tests/test_server.py) остаются.
3. **Риск "большого взрыва".** 7 фаз, каждая 300-700 строк дельты, каждая самостоятельно мержабельна. Нет шага, где половина архитектур сломана.
4. **Риск недооценки объёма.** Ориентировочно: Ф1 — 1 день, Ф2 — 1 день, Ф3 — 3-4 дня, Ф4 — 3-4 дня, Ф5 — 3-5 дней, Ф6 — 1-2 дня, Ф7 — 1 день. Суммарно ~2-3 недели фокусной работы.

## Критерии приёмки

- [ ] Все тесты ([test_dsl](tests/test_dsl.py), [test_e2e](tests/test_e2e.py), [test_new_architectures](tests/test_new_architectures.py), [test_server](tests/test_server.py), новый test_golden) зелёные.
- [ ] `dsl.py` < 1500 строк (сейчас 2867; ~50% уходит в IR + lowering + примитивы).
- [ ] Количество `isinstance(layer, X)` веток в одном файле ≤ 20 (сейчас 55+).
- [ ] В `.tex` выхлопе нет дробных cm с 4+ знаками, нет дублирующих `{Output}`, `Input` всегда цепляется к non-section узлу.
- [ ] Новая архитектура (e.g., Mixtral-style MoE) пишется пользователем в 5-10 строк без необходимости добавлять ветку в `dsl.py`.
- [ ] Единая цветовая семантика: `grep -h "fill=clr" *.tex | sort -u` показывает, что каждый цвет соответствует ровно одной семантической роли.
