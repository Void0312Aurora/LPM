"use strict";

const state = {
  bootstrap: null,
  current: null,
  trajectoryId: null,
  eventId: null,
  decision: "accepted",
  dirty: false,
  loading: false,
  toastTimer: null,
  picker: null,
  sourceAlignmentOpen: false,
};

const $ = (selector) => document.querySelector(selector);

function make(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== "") element.textContent = text;
  return element;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(`服务器返回非 JSON 响应 (${response.status})`);
  }
  if (!response.ok || !payload.ok) {
    const error = new Error(payload.error || `请求失败 (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function setSaveState(text, mode = "") {
  const element = $("#save-state");
  element.textContent = text;
  element.className = `save-state ${mode}`.trim();
}

function markDirty() {
  if (state.loading || !state.current) return;
  state.dirty = true;
  setSaveState("有未保存修改", "dirty");
}

function showToast(message, isError = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast visible${isError ? " error" : ""}`;
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    toast.className = "toast";
  }, 3500);
}

function formatSlot(value) {
  if (Array.isArray(value)) return value.join(", ");
  return value == null ? "" : String(value);
}

function parseList(value) {
  const seen = new Set();
  return String(value || "")
    .replaceAll("，", ",")
    .replaceAll("、", ",")
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter((item) => {
      if (!item || seen.has(item)) return false;
      seen.add(item);
      return true;
    });
}

function parseSlot(value) {
  const items = parseList(value);
  if (items.length === 0) return null;
  if (items.length === 1) return items[0];
  return items;
}

function asValueList(value) {
  if (Array.isArray(value)) return [...new Set(value.map((item) => String(item).trim()).filter(Boolean))];
  if (value == null || value === "") return [];
  return [String(value).trim()].filter(Boolean);
}

function resolveElement(target) {
  return typeof target === "string" ? $(target) : target;
}

function selectionValues(target) {
  const element = resolveElement(target);
  if (!element?.dataset.values) return [];
  try {
    return asValueList(JSON.parse(element.dataset.values));
  } catch (_error) {
    return [];
  }
}

function selectionSlot(target) {
  const values = selectionValues(target);
  if (values.length === 0) return null;
  if (values.length === 1) return values[0];
  return values;
}

function renderSelectionField(target, config) {
  const element = resolveElement(target);
  const values = asValueList(config.values);
  const lockedValues = asValueList(config.lockedValues);
  const disabled = Boolean(config.disabled);
  element.replaceChildren();
  element.dataset.values = JSON.stringify(values);
  element.dataset.lockedValues = JSON.stringify(lockedValues);
  element.dataset.pickerKind = config.kind || "characters";
  element.dataset.pickerLabel = config.label || "选项";
  element.dataset.disabled = disabled ? "true" : "false";
  element.classList.toggle("disabled", disabled);

  const chips = make("div", "selection-chips");
  if (!values.length) {
    chips.append(make("span", "selection-placeholder", config.emptyText || "未选择（null）"));
  } else {
    values.forEach((value) => {
      const chip = make("span", "selection-chip");
      chip.append(make("span", "", value));
      if (!disabled && !lockedValues.includes(value)) {
        const remove = make("button", "selection-chip-remove", "×");
        remove.type = "button";
        remove.setAttribute("aria-label", `移除 ${config.label || "选项"} ${value}`);
        remove.addEventListener("click", () => {
          renderSelectionField(element, {
            ...config,
            values: selectionValues(element).filter((item) => item !== value),
          });
          markDirty();
        });
        chip.append(remove);
      }
      chips.append(chip);
    });
  }
  element.append(chips);

  const open = make(
    "button",
    "selection-open",
    disabled ? "源字段锁定" : `选择${config.label || "选项"}`,
  );
  open.type = "button";
  open.disabled = disabled;
  open.setAttribute("aria-label", disabled ? `${config.label || "选项"} 已锁定` : `选择 ${config.label || "选项"}`);
  open.addEventListener("click", () => openPresetPicker(element));
  element.append(open);
}

function updateSelectionFieldValues(target, values) {
  const element = resolveElement(target);
  renderSelectionField(element, {
    values,
    lockedValues: JSON.parse(element.dataset.lockedValues || "[]"),
    kind: element.dataset.pickerKind,
    label: element.dataset.pickerLabel,
    disabled: element.dataset.disabled === "true",
  });
}

function contextCharacterNames() {
  const names = new Set();
  const add = (value) => asValueList(value).forEach((item) => names.add(item));
  for (const event of state.current?.context || []) {
    add(event.agent);
    add(event.target);
    add(event.participants);
  }
  add(state.current?.evidence?.canonical?.speaker_norm);
  add(state.current?.evidence?.canonical?.speaker_raw);
  for (const aligned of state.current?.evidence?.aligned || []) {
    add(aligned.zh_speaker);
  }
  return names;
}

function isSelectableCharacterName(name) {
  const normalized = String(name || "").trim();
  if (!normalized) return false;
  if (/^[\p{P}\p{S}\p{N}\s]+$/u.test(normalized)) return false;
  if (/^\$\(/u.test(normalized)) return false;
  return true;
}

function availablePresetOptions(kind, selectedValues = []) {
  if (kind === "flags") {
    const values = new Set([
      ...state.bootstrap.flag_suggestions,
      ...(state.current?.policy?.base_flags || []),
      ...selectedValues,
    ]);
    return [...values].sort().map((name) => ({ name, count: null, contextual: false }));
  }
  const contextual = contextCharacterNames();
  const metadata = new Map(state.bootstrap.characters.map((item) => [item.name, Number(item.count || 0)]));
  [...selectedValues, ...contextual].forEach((name) => {
    if (name && !metadata.has(name)) metadata.set(name, 0);
  });
  return [...metadata.entries()]
    .filter(([name]) => isSelectableCharacterName(name) || selectedValues.includes(name))
    .map(([name, count]) => ({ name, count, contextual: contextual.has(name) }))
    .sort((left, right) => {
      if (left.contextual !== right.contextual) return left.contextual ? -1 : 1;
      if (left.count !== right.count) return right.count - left.count;
      return left.name.localeCompare(right.name, "zh-CN");
    });
}

function updatePresetPickerCount() {
  if (!state.picker) return;
  $("#preset-picker-count").textContent = `已选 ${state.picker.selected.size}`;
}

function isDefaultCharacterOption(option) {
  const name = option.name.trim();
  if (!name || option.count < 10) return false;
  if (/^[\p{P}\p{S}\p{N}\s]+$/u.test(name)) return false;
  if (/[·・]/u.test(name)) return false;
  return true;
}

function renderPresetPickerOptions() {
  if (!state.picker) return;
  const query = $("#preset-picker-search").value.trim().toLowerCase();
  const container = $("#preset-picker-options");
  container.replaceChildren();
  const matched = availablePresetOptions(state.picker.kind, [...state.picker.selected]).filter((option) =>
    option.name.toLowerCase().includes(query),
  );
  let options = matched;
  if (state.picker.kind === "characters") {
    if (query) {
      options = matched.slice(0, 200);
      $("#preset-picker-hint").textContent = `匹配 ${matched.length} 项，最多显示 200 项`;
    } else {
      options = matched
        .filter(
          (option) =>
            option.contextual ||
            isDefaultCharacterOption(option) ||
            state.picker.selected.has(option.name) ||
            state.picker.locked.has(option.name),
        )
        .slice(0, 100);
      $("#preset-picker-hint").textContent = `默认显示上下文及受控常用角色 ${options.length} 项；搜索可查全部`;
    }
  } else {
    $("#preset-picker-hint").textContent = `闭集 flag 共 ${matched.length} 项`;
  }
  if (!options.length) {
    container.append(make("p", "preset-empty", "没有匹配的预制选项"));
    return;
  }
  options.forEach((option) => {
    const label = make("label", `preset-option${option.contextual ? " contextual" : ""}`);
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = option.name;
    input.checked = state.picker.selected.has(option.name);
    input.disabled = state.picker.locked.has(option.name);
    input.addEventListener("change", () => {
      if (input.checked) state.picker.selected.add(option.name);
      else state.picker.selected.delete(option.name);
      updatePresetPickerCount();
    });
    const text = make("span", "preset-option-name", option.name);
    label.append(input, text);
    if (option.contextual) label.append(make("span", "preset-context-badge", "上下文"));
    if (option.count != null) label.append(make("span", "preset-option-count", String(option.count)));
    container.append(label);
  });
}

function openPresetPicker(target) {
  const element = resolveElement(target);
  if (element.dataset.disabled === "true") return;
  state.picker = {
    target: element,
    kind: element.dataset.pickerKind || "characters",
    label: element.dataset.pickerLabel || "选项",
    selected: new Set(selectionValues(element)),
    locked: new Set(JSON.parse(element.dataset.lockedValues || "[]")),
  };
  $("#preset-picker-title").textContent = `选择 ${state.picker.label}`;
  $("#preset-picker-search").value = "";
  $("#preset-picker-modal").classList.remove("hidden");
  $("#preset-picker-modal").setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
  updatePresetPickerCount();
  renderPresetPickerOptions();
  $("#preset-picker-search").focus();
}

function closePresetPicker() {
  state.picker = null;
  $("#preset-picker-modal").classList.add("hidden");
  $("#preset-picker-modal").setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
}

function applyPresetPicker() {
  if (!state.picker) return;
  const { target, selected } = state.picker;
  updateSelectionFieldValues(target, [...selected]);
  closePresetPicker();
  markDirty();
}

function clearPresetPicker() {
  if (!state.picker) return;
  state.picker.selected = new Set(state.picker.locked);
  renderPresetPickerOptions();
  updatePresetPickerCount();
}

function actionTypeFor(type) {
  if (type === "utterance") return "speak";
  if (type === "action") return "action";
  return "other";
}

function statusLabel(status) {
  return {
    accepted: "已验收",
    needs_review: "待复核",
    skipped: "已跳过",
    unannotated: "未标注",
  }[status || "unannotated"];
}

function countsTotal(counts = {}) {
  return Object.values(counts).reduce((total, value) => total + Number(value || 0), 0);
}

function updateGlobalCounts(counts = {}) {
  $("#global-accepted").textContent = counts.accepted || 0;
  $("#global-review").textContent = counts.needs_review || 0;
  $("#global-skipped").textContent = counts.skipped || 0;
}

function trajectoryMatchesFilter(trajectory, filter) {
  if (filter === "started") return trajectory.annotated_count > 0;
  if (filter === "unstarted") return trajectory.annotated_count === 0;
  if (filter === "needs_review") return (trajectory.status_counts.needs_review || 0) > 0;
  return true;
}

function renderTrajectories() {
  if (!state.bootstrap) return;
  const query = $("#trajectory-search").value.trim().toLowerCase();
  const filter = $("#trajectory-filter").value;
  const trajectories = state.bootstrap.trajectories.filter((trajectory) => {
    const haystack = `${trajectory.trajectory_id} ${trajectory.scene || ""}`.toLowerCase();
    return haystack.includes(query) && trajectoryMatchesFilter(trajectory, filter);
  });
  $("#trajectory-count").textContent = trajectories.length;
  const list = $("#trajectory-list");
  list.replaceChildren();
  for (const trajectory of trajectories) {
    const button = make("button", `trajectory-item${trajectory.trajectory_id === state.trajectoryId ? " active" : ""}`);
    button.type = "button";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", trajectory.trajectory_id === state.trajectoryId ? "true" : "false");

    const top = make("div", "trajectory-item-top");
    top.append(make("span", "trajectory-id", trajectory.trajectory_id));
    const review = trajectory.status_counts.needs_review || 0;
    if (review) top.append(make("span", "status-badge needs_review", `${review} 复核`));
    button.append(top);
    button.append(make("div", "trajectory-scene", trajectory.scene || "(no scene)"));

    const bottom = make("div", "trajectory-item-bottom");
    bottom.append(make("span", "", `${trajectory.annotated_count}/${trajectory.count}`));
    const mini = make("span", "mini-progress");
    const fill = make("span");
    fill.style.width = `${trajectory.count ? (trajectory.annotated_count / trajectory.count) * 100 : 0}%`;
    mini.append(fill);
    bottom.append(mini);
    button.append(bottom);

    button.addEventListener("click", () => loadItem(trajectory.trajectory_id, null));
    list.append(button);
  }
}

function currentTrajectoryMeta() {
  return state.bootstrap?.trajectories.find((item) => item.trajectory_id === state.trajectoryId) || null;
}

function updateTrajectoryCounts(counts) {
  const trajectory = currentTrajectoryMeta();
  if (!trajectory) return;
  trajectory.status_counts = { ...counts };
  trajectory.annotated_count = countsTotal(counts);
  trajectory.unannotated_count = Math.max(0, trajectory.count - trajectory.annotated_count);
  renderTrajectories();
}

function renderContext(item) {
  const list = $("#context-list");
  list.replaceChildren();
  $("#context-window-summary").textContent = `显示 ${item.context.length} / ${item.total} 条 · 点击切换`;
  let currentButton = null;
  for (const event of item.context) {
    const current = Number(event.id) === Number(item.event.id);
    const button = make("button", `context-event${current ? " current" : ""}`);
    button.type = "button";
    button.append(make("span", "context-id", `#${event.id}`));
    const meta = event.type || event.kind || "unknown";
    const agent = formatSlot(event.agent) || "—";
    button.append(make("span", "context-meta", `${meta} · ${agent}`));
    button.append(make("span", "context-text", event.text || `[${event.kind}]`));
    button.append(make("span", `context-status-dot ${event.annotation_status || ""}`));
    button.addEventListener("click", () => loadItem(state.trajectoryId, Number(event.id)));
    list.append(button);
    if (current) currentButton = button;
  }
  window.requestAnimationFrame(() => {
    if (!currentButton) return;
    const listRect = list.getBoundingClientRect();
    const itemRect = currentButton.getBoundingClientRect();
    if (itemRect.top < listRect.top || itemRect.bottom > listRect.bottom) {
      list.scrollTop += itemRect.top - listRect.top - (list.clientHeight - itemRect.height) / 2;
    }
  });
}

function alignmentLine(label, value) {
  const line = make("p", "alignment-line");
  line.append(make("strong", "", label));
  line.append(document.createTextNode(value == null || value === "" ? "—" : String(value)));
  return line;
}

function renderSourceAlignment(item) {
  const source = item.event.source || {};
  const alignedRows = item.evidence?.aligned || [];
  $("#source-alignment-button").classList.toggle("hidden", alignedRows.length === 0);
  $("#source-alignment-locator").textContent = `${source.dataset || "?"} · ${source.scene || "?"} · ${source.path || "?"}`;
  const container = $("#source-alignment-content");
  container.replaceChildren();
  alignedRows.forEach((aligned, index) => {
    const card = make("article", "alignment-card");
    card.append(make("div", "alignment-label", `对齐记录 ${index + 1}`));
    card.append(alignmentLine("日文", `${aligned.speaker || "旁白"}: ${aligned.text || ""}`));
    card.append(alignmentLine("中文", `${aligned.zh_speaker || "旁白"}: ${aligned.zh_text || ""}`));
    card.append(alignmentLine("英文", aligned.en_text));
    container.append(card);
  });
}

function openSourceAlignment() {
  if (!state.current?.evidence?.aligned?.length) return;
  state.sourceAlignmentOpen = true;
  const modal = $("#source-alignment-modal");
  modal.classList.remove("hidden");
  modal.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
  $("#source-alignment-close").focus();
}

function closeSourceAlignment() {
  state.sourceAlignmentOpen = false;
  const modal = $("#source-alignment-modal");
  modal.classList.add("hidden");
  modal.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
  $("#source-alignment-button").focus();
}

function setSelectedType(type) {
  document.querySelectorAll(".type-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.type === type);
    button.setAttribute("aria-checked", button.dataset.type === type ? "true" : "false");
  });
  $("#action-type-select").value = actionTypeFor(type);
  if (type === "monologue") updateSelectionFieldValues("#target-picker", []);
}

function selectedType() {
  return document.querySelector(".type-button.active")?.dataset.type || "other";
}

function renderTypeButtons(item) {
  const container = $("#type-buttons");
  container.replaceChildren();
  const lockedType = item.policy.locked_fields.type;
  for (const type of state.bootstrap.normal_types) {
    const button = make("button", `type-button${item.patch.type === type ? " active" : ""}`, type);
    button.type = "button";
    button.dataset.type = type;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", item.patch.type === type ? "true" : "false");
    button.disabled = Boolean(lockedType);
    button.addEventListener("click", () => {
      setSelectedType(type);
      markDirty();
    });
    container.append(button);
  }
}

function renderFlags(item) {
  const baseFlags = new Set(item.policy.base_flags || []);
  const selected = new Set(item.patch.flag || []);
  const known = new Set([...state.bootstrap.flag_suggestions, ...baseFlags, ...selected]);
  const container = $("#flag-options");
  container.replaceChildren();
  [...known].sort().forEach((flag, index) => {
    const label = make("label", "flag-option");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = flag;
    input.id = `flag-${index}`;
    input.checked = selected.has(flag);
    input.disabled = baseFlags.has(flag);
    input.addEventListener("change", markDirty);
    label.append(input, make("span", "", flag));
    container.append(label);
  });
}

function renderChoiceOptions(item) {
  const container = $("#choice-options");
  container.replaceChildren();
  const sourceOptions = item.event.choice?.options || {};
  const patchOptions = item.patch.choice_options || {};
  for (const [index, sourceOption] of Object.entries(sourceOptions)) {
    const patch = patchOptions[index] || sourceOption;
    const card = make("article", "choice-option-card");
    card.dataset.optionIndex = index;
    card.append(make("h4", "", `Option ${index}`));
    card.append(make("p", "choice-option-text", sourceOption.text || "(empty option)"));
    const grid = make("div", "choice-option-grid");

    const typeLabel = make("label", "", "Type");
    const typeSelect = document.createElement("select");
    typeSelect.className = "option-type";
    [...state.bootstrap.normal_types, "invalid"].forEach((type) => {
      const option = document.createElement("option");
      option.value = type;
      option.textContent = type;
      option.selected = patch.type === type;
      typeSelect.append(option);
    });
    typeLabel.append(typeSelect);
    grid.append(typeLabel);

    const validLabel = make("label", "", "Valid");
    const validSelect = document.createElement("select");
    validSelect.className = "option-valid";
    for (const [value, label] of [["true", "true"], ["false", "false"]]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      option.selected = Boolean(patch.valid) === (value === "true");
      validSelect.append(option);
    }
    validLabel.append(validSelect);
    grid.append(validLabel);

    card.append(grid);
    const selections = make("div", "choice-option-selections");
    const selectionConfigs = [
      ["Agent", "option-agent-picker", "characters", patch.agent, []],
      ["Target", "option-target-picker", "characters", patch.target, []],
      ["Participants", "option-participants-picker", "characters", patch.participants, []],
      ["Flags", "option-flags-picker", "flags", patch.flag, sourceOption.flag || []],
    ];
    for (const [labelText, className, kind, values, lockedValues] of selectionConfigs) {
      const wrapper = make("div", "field choice-selection-field");
      wrapper.append(make("span", "", labelText));
      const picker = make("div", `selection-field ${className}`);
      picker.setAttribute("role", "group");
      picker.setAttribute("aria-label", `Option ${index} ${labelText}`);
      renderSelectionField(picker, {
        label: `Option ${index} ${labelText}`,
        kind,
        values,
        lockedValues,
        emptyText: labelText === "Target" ? "未选择（null）" : "未选择",
      });
      wrapper.append(picker);
      selections.append(wrapper);
    }
    card.append(selections);
    typeSelect.addEventListener("change", () => {
      if (typeSelect.value === "monologue") updateSelectionFieldValues(card.querySelector(".option-target-picker"), []);
      markDirty();
    });
    validSelect.addEventListener("change", markDirty);
    container.append(card);
  }
}

function setDecision(status, dirty = true) {
  state.decision = status;
  document.querySelectorAll(".decision-button").forEach((button) => {
    const active = button.dataset.status === status;
    button.classList.toggle("active", active);
    button.setAttribute("aria-checked", active ? "true" : "false");
  });
  if (dirty) markDirty();
}

function renderEditor(item) {
  const kind = item.event.kind;
  $("#normal-fields").classList.toggle("hidden", kind !== "normal");
  $("#choice-fields").classList.toggle("hidden", kind !== "choice");
  $("#lock-notice").classList.toggle("hidden", !item.policy.speaker_locked);
  $("#stale-notice").classList.toggle("hidden", !item.stale_annotation);

  if (kind === "normal") {
    renderTypeButtons(item);
    $("#action-type-select").value = item.patch.action_type || actionTypeFor(item.patch.type);
    renderSelectionField("#agent-picker", {
      label: "Agent",
      kind: "characters",
      values: item.patch.agent,
      lockedValues: item.policy.speaker_locked ? asValueList(item.patch.agent) : [],
      disabled: item.policy.speaker_locked,
      emptyText: "未选择 Agent",
    });
    renderSelectionField("#target-picker", {
      label: "Target",
      kind: "characters",
      values: item.patch.target,
      emptyText: "未选择（null）",
    });
    renderSelectionField("#participants-picker", {
      label: "Participants",
      kind: "characters",
      values: item.patch.participants,
      emptyText: "未选择 Participants",
    });
  } else if (kind === "choice") {
    renderChoiceOptions(item);
  }
  $("#valid-input").checked = Boolean(item.patch.valid);
  $("#valid-input").disabled = kind === "invalid";
  renderFlags(item);
  $("#note-input").value = item.annotation?.note || "";
  setDecision(item.annotation?.status || "accepted", false);
  hideValidationErrors();
}

function renderItem() {
  const item = state.current;
  if (!item) return;
  const event = item.event;
  $("#scene-name").textContent = `${item.trajectory.trajectory_id} · ${item.trajectory.scene || ""}`;
  $("#event-title").textContent = `事件 #${event.id} · ${event.kind}${event.type ? ` / ${event.type}` : ""}`;
  $("#event-id-input").value = event.id;
  const status = item.annotation?.status || "unannotated";
  const badge = $("#event-status-badge");
  badge.textContent = statusLabel(status);
  badge.className = `status-badge ${status}`;

  $("#previous-button").disabled = item.navigation.previous_id == null;
  $("#next-button").disabled = item.navigation.next_id == null;
  $("#next-unannotated-button").disabled = item.navigation.next_unannotated_id == null;

  const annotated = countsTotal(item.trajectory_status_counts);
  const percent = item.total ? (annotated / item.total) * 100 : 0;
  $("#trajectory-progress").style.width = `${percent}%`;
  $("#trajectory-progress-text").textContent = `${annotated} 已标注 / ${item.total} 总计 · 当前位置 ${item.position + 1}`;

  renderContext(item);
  renderSourceAlignment(item);
  renderEditor(item);
}

function canLeaveCurrent() {
  if (!state.dirty) return true;
  return window.confirm("当前事件有未保存修改，仍要离开吗？");
}

async function loadItem(trajectoryId, eventId, { force = false } = {}) {
  if (state.loading) return;
  if (!force && !canLeaveCurrent()) return;
  state.loading = true;
  setSaveState("加载中…");
  try {
    const context = Number($("#context-radius").value || state.bootstrap?.default_context || 16);
    const params = new URLSearchParams({ trajectory: trajectoryId, context: String(context) });
    if (eventId != null) params.set("event_id", String(eventId));
    const payload = await fetchJSON(`/api/item?${params}`);
    state.current = payload.item;
    state.trajectoryId = payload.item.trajectory.trajectory_id;
    state.eventId = Number(payload.item.event.id);
    state.dirty = false;
    renderTrajectories();
    renderItem();
    const url = new URL(window.location.href);
    url.searchParams.set("trajectory", state.trajectoryId);
    url.searchParams.set("event_id", String(state.eventId));
    window.history.replaceState(null, "", url);
    setSaveState(state.current.annotation ? `revision ${state.current.annotation.revision}` : "未保存");
  } catch (error) {
    showToast(error.message, true);
    setSaveState("加载失败");
  } finally {
    state.loading = false;
  }
}

function collectFlags() {
  const selected = [...document.querySelectorAll("#flag-options input:checked")].map((input) => input.value);
  return [...new Set(selected)].sort();
}

function collectChoiceOptions() {
  const options = {};
  document.querySelectorAll(".choice-option-card").forEach((card) => {
    const type = card.querySelector(".option-type").value;
    let target = selectionSlot(card.querySelector(".option-target-picker"));
    if (type === "monologue") target = null;
    options[card.dataset.optionIndex] = {
      type,
      action_type: actionTypeFor(type),
      agent: selectionSlot(card.querySelector(".option-agent-picker")),
      target,
      participants: selectionValues(card.querySelector(".option-participants-picker")),
      valid: card.querySelector(".option-valid").value === "true",
      flag: selectionValues(card.querySelector(".option-flags-picker")),
    };
  });
  return options;
}

function collectPatch() {
  const kind = state.current.event.kind;
  const common = {
    valid: $("#valid-input").checked,
    flag: collectFlags(),
  };
  if (kind === "normal") {
    const type = selectedType();
    return {
      ...common,
      type,
      action_type: actionTypeFor(type),
      agent: selectionSlot("#agent-picker"),
      target: type === "monologue" ? null : selectionSlot("#target-picker"),
      participants: selectionValues("#participants-picker"),
    };
  }
  if (kind === "choice") {
    return { ...common, choice_options: collectChoiceOptions() };
  }
  return common;
}

function showValidationErrors(errors) {
  const container = $("#validation-errors");
  container.replaceChildren();
  container.append(make("strong", "", "无法保存："));
  const list = make("ul");
  for (const error of errors || []) {
    const path = Array.isArray(error.path) ? error.path.join(".") : "";
    list.append(make("li", "", `${path ? `${path}: ` : ""}${error.message}`));
  }
  container.append(list);
  container.classList.remove("hidden");
}

function hideValidationErrors() {
  $("#validation-errors").classList.add("hidden");
  $("#validation-errors").replaceChildren();
}

function setSaveButtonsDisabled(disabled) {
  $("#save-button").disabled = disabled;
  $("#save-next-button").disabled = disabled;
}

function clientValidationErrors(patch) {
  const errors = [];
  const note = $("#note-input").value.trim();
  if (state.decision === "needs_review" && !note) {
    errors.push({ path: ["note"], message: "待复核必须填写人工备注" });
  }
  if (state.decision === "accepted" && state.current.event.kind === "normal") {
    if (["utterance", "action", "monologue"].includes(patch.type) && !patch.agent) {
      errors.push({ path: ["patch", "agent"], message: `${patch.type} 必须指定 agent` });
    }
  }
  if (state.decision === "accepted" && !patch.valid && patch.flag.length === 0) {
    errors.push({ path: ["patch", "flag"], message: "不可用事件至少需要一个 flag" });
  }
  return errors;
}

async function saveAnnotation({ advance = false } = {}) {
  if (!state.current || state.loading) return;
  hideValidationErrors();
  const patch = collectPatch();
  const localErrors = clientValidationErrors(patch);
  if (localErrors.length) {
    showValidationErrors(localErrors);
    showToast("请先补全人工判定", true);
    return;
  }
  setSaveButtonsDisabled(true);
  setSaveState("保存中…");
  const nextId = state.current.navigation.next_id;
  try {
    const payload = await fetchJSON("/api/annotations", {
      method: "POST",
      body: JSON.stringify({
        trajectory: state.trajectoryId,
        event_id: state.eventId,
        status: state.decision,
        annotator: state.bootstrap.annotator,
        note: $("#note-input").value,
        patch,
        expected_revision: state.current.annotation?.revision || 0,
      }),
    });
    state.dirty = false;
    updateGlobalCounts(payload.global_status_counts);
    updateTrajectoryCounts(payload.trajectory_status_counts);
    showToast(`事件 #${state.eventId} 已保存为 ${statusLabel(state.decision)}`);
    if (advance && nextId != null) {
      await loadItem(state.trajectoryId, nextId, { force: true });
    } else {
      await loadItem(state.trajectoryId, state.eventId, { force: true });
    }
  } catch (error) {
    if (error.status === 422) {
      showValidationErrors(error.payload.errors);
      showToast("标注未通过校验", true);
    } else if (error.status === 409) {
      showToast("该事件已在其他页面更新，正在重新加载", true);
      state.dirty = false;
      await loadItem(state.trajectoryId, state.eventId, { force: true });
    } else {
      showToast(error.message, true);
    }
    setSaveState("保存失败", "dirty");
  } finally {
    setSaveButtonsDisabled(false);
  }
}

async function exportSnapshot() {
  const button = $("#export-button");
  button.disabled = true;
  try {
    const payload = await fetchJSON("/api/export", { method: "POST", body: "{}" });
    const summary = payload.summary;
    showToast(
      `已导出 ${summary.accepted_event_count} 条验收事件，${summary.needs_review_count} 条待复核到 ${payload.output_dir}`,
    );
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

function resetToBaseline() {
  if (!state.current) return;
  state.current.patch = structuredClone(state.current.default_patch);
  renderEditor(state.current);
  markDirty();
}

function bindEvents() {
  $("#trajectory-search").addEventListener("input", renderTrajectories);
  $("#trajectory-filter").addEventListener("change", renderTrajectories);
  $("#context-radius").addEventListener("change", () => {
    if (state.current) loadItem(state.trajectoryId, state.eventId);
  });
  $("#previous-button").addEventListener("click", () => {
    const id = state.current?.navigation.previous_id;
    if (id != null) loadItem(state.trajectoryId, id);
  });
  $("#next-button").addEventListener("click", () => {
    const id = state.current?.navigation.next_id;
    if (id != null) loadItem(state.trajectoryId, id);
  });
  $("#next-unannotated-button").addEventListener("click", () => {
    const id = state.current?.navigation.next_unannotated_id;
    if (id != null) loadItem(state.trajectoryId, id);
  });
  $("#event-id-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      loadItem(state.trajectoryId, Number(event.currentTarget.value));
    }
  });
  $("#annotation-form").addEventListener("submit", (event) => {
    event.preventDefault();
    saveAnnotation();
  });
  $("#annotation-form").addEventListener("input", (event) => {
    if (!event.target.closest(".decision-button")) markDirty();
  });
  $("#save-next-button").addEventListener("click", () => saveAnnotation({ advance: true }));
  $("#reset-button").addEventListener("click", resetToBaseline);
  $("#export-button").addEventListener("click", exportSnapshot);
  $("#source-alignment-button").addEventListener("click", openSourceAlignment);
  $("#source-alignment-close").addEventListener("click", closeSourceAlignment);
  $("#source-alignment-done").addEventListener("click", closeSourceAlignment);
  $("#source-alignment-modal").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeSourceAlignment();
  });
  $("#preset-picker-search").addEventListener("input", renderPresetPickerOptions);
  $("#preset-picker-close").addEventListener("click", closePresetPicker);
  $("#preset-picker-cancel").addEventListener("click", closePresetPicker);
  $("#preset-picker-apply").addEventListener("click", applyPresetPicker);
  $("#preset-picker-clear").addEventListener("click", clearPresetPicker);
  $("#preset-picker-modal").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closePresetPicker();
  });
  document.querySelectorAll(".decision-button").forEach((button) => {
    button.addEventListener("click", () => setDecision(button.dataset.status));
  });
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
  window.addEventListener("keydown", (event) => {
    if (state.sourceAlignmentOpen) {
      if (event.key === "Escape") closeSourceAlignment();
      return;
    }
    if (state.picker) {
      if (event.key === "Escape") closePresetPicker();
      return;
    }
    const modifier = event.ctrlKey || event.metaKey;
    if (modifier && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveAnnotation();
    } else if (modifier && event.key === "Enter") {
      event.preventDefault();
      saveAnnotation({ advance: true });
    } else if (event.altKey && event.key === "ArrowLeft") {
      event.preventDefault();
      const id = state.current?.navigation.previous_id;
      if (id != null) loadItem(state.trajectoryId, id);
    } else if (event.altKey && event.key === "ArrowRight") {
      event.preventDefault();
      const id = state.current?.navigation.next_id;
      if (id != null) loadItem(state.trajectoryId, id);
    }
  });
}

async function initialize() {
  bindEvents();
  try {
    const payload = await fetchJSON("/api/bootstrap");
    state.bootstrap = payload;
    $("#annotator-name").textContent = payload.annotator;
    $("#context-radius").value = String(payload.default_context);
    updateGlobalCounts(payload.global_status_counts);
    renderTrajectories();
    const params = new URLSearchParams(window.location.search);
    const trajectory = params.get("trajectory") || payload.trajectories[0]?.trajectory_id;
    const eventId = params.has("event_id") ? Number(params.get("event_id")) : null;
    if (!trajectory) throw new Error("数据集中没有可用轨迹");
    await loadItem(trajectory, Number.isFinite(eventId) ? eventId : null, { force: true });
    if (payload.history_load_errors.length) {
      showToast(`历史日志有 ${payload.history_load_errors.length} 条无法读取`, true);
    }
  } catch (error) {
    showToast(error.message, true);
    setSaveState("初始化失败");
  }
}

window.addEventListener("DOMContentLoaded", initialize);
