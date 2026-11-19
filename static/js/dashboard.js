(() => {
  const root = document.querySelector("[data-dashboard]");
  if (!root) return;
  const categoryNames = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];
  const colors = ["#6e9a72", "#8d7dc0", "#d4a84e", "#6e91b4", "#59aaa0", "#9aa39c"];
  const label = (value) => value === "uncertain" ? "Not sure yet" : value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

  async function load(period = "weekly") {
    const response = await fetch(`/api/stats?period=${period}`);
    if (!response.ok) throw new Error("Dashboard statistics could not be loaded.");
    const data = await response.json();
    root.querySelector("[data-total]").textContent = data.total_scans;
    root.querySelector("[data-recyclable]").textContent = data.recyclable_scans;
    root.querySelector("[data-category-count]").textContent = Object.keys(data.categories).length;
    drawActivity(data.daily);
    drawCategories(data.categories);
    drawRecent(data.recent);
    loadEvaluation().catch((error) => {
      const empty = root.querySelector("[data-evaluation-empty]");
      empty.textContent = error.message;
      empty.hidden = false;
      root.querySelector("[data-evaluation-content]").hidden = true;
    });
  }

  async function loadEvaluation() {
    const response = await fetch("/api/model-metrics");
    if (!response.ok) throw new Error("Model evaluation could not be loaded.");
    const report = await response.json();
    const empty = root.querySelector("[data-evaluation-empty]");
    const content = root.querySelector("[data-evaluation-content]");
    if (!report.available) {
      empty.hidden = false;
      content.hidden = true;
      return;
    }
    empty.hidden = true;
    content.hidden = false;
    root.querySelector("[data-evaluation-samples]").textContent = `${report.test_samples} test images`;
    ["accuracy", "macro_precision", "macro_recall", "macro_f1_score"].forEach((key) => {
      root.querySelector(`[data-metric="${key}"]`).textContent = `${(report[key] * 100).toFixed(1)}%`;
    });
    const scores = document.createElement("table");
    scores.className = "confusion-table";
    const scoreHead = document.createElement("thead");
    const scoreHeader = document.createElement("tr");
    ["Category", "Precision", "Recall", "F1"].forEach((title) => {
      const cell = document.createElement("th");
      cell.textContent = title;
      scoreHeader.append(cell);
    });
    scoreHead.append(scoreHeader);
    const scoreBody = document.createElement("tbody");
    Object.entries(report.per_class).forEach(([name, values]) => {
      const row = document.createElement("tr");
      const category = document.createElement("th");
      category.scope = "row";
      category.textContent = label(name);
      row.append(category);
      ["precision", "recall", "f1_score"].forEach((key) => {
        const cell = document.createElement("td");
        cell.textContent = `${(values[key] * 100).toFixed(1)}%`;
        row.append(cell);
      });
      scoreBody.append(row);
    });
    scores.append(scoreHead, scoreBody);
    root.querySelector("[data-per-class]").replaceChildren(scores);

    const table = document.createElement("table");
    table.className = "confusion-table";
    const header = document.createElement("tr");
    header.append(document.createElement("th"));
    report.class_order.forEach((name) => {
      const cell = document.createElement("th");
      cell.textContent = label(name);
      header.append(cell);
    });
    const head = document.createElement("thead");
    head.append(header);
    const body = document.createElement("tbody");
    report.confusion_matrix.forEach((values, rowIndex) => {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = label(report.class_order[rowIndex]);
      row.append(name);
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      body.append(row);
    });
    table.append(head, body);
    root.querySelector("[data-confusion-matrix]").replaceChildren(table);
  }

  function drawActivity(rows) {
    const canvas = root.querySelector("[data-activity-chart]");
    const empty = root.querySelector("[data-chart-empty]");
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(300, rect.width);
    const height = 220;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, height);
    if (!rows.length) {
      empty.hidden = false;
      return;
    }
    empty.hidden = true;
    const now = new Date();
    const today = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const countDays = root.querySelector("[data-period]").value === "daily" ? 1 : 7;
    const days = Array.from({ length: countDays }, (_, index) => {
      const date = new Date(today);
      date.setUTCDate(today.getUTCDate() - (countDays - index - 1));
      const key = `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
      return { key, label: date.toLocaleDateString(undefined, { weekday: "short", timeZone: "UTC" }), count: rows.find((row) => row.day === key)?.count || 0 };
    });
    const max = Math.max(1, ...days.map((day) => day.count));
    const gap = width / days.length;
    const barWidth = Math.min(34, gap * 0.42);
    days.forEach((day, index) => {
      const x = gap * index + (gap - barWidth) / 2;
      const barHeight = (day.count / max) * 137;
      ctx.fillStyle = "#edf2ec";
      ctx.beginPath();
      ctx.roundRect(x, 24, barWidth, 142, 8);
      ctx.fill();
      if (day.count) {
        ctx.fillStyle = "#358653";
        ctx.beginPath();
        ctx.roundRect(x, 166 - barHeight, barWidth, barHeight, 8);
        ctx.fill();
      }
      ctx.fillStyle = "#879089";
      ctx.font = "12px DM Sans, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(day.label, gap * index + gap / 2, 198);
      if (day.count) {
        ctx.fillStyle = "#293b30";
        ctx.font = "600 12px DM Sans, sans-serif";
        ctx.fillText(String(day.count), gap * index + gap / 2, 157 - barHeight);
      }
    });
  }

  function drawCategories(categories) {
    const total = Object.values(categories).reduce((sum, count) => sum + count, 0);
    const chart = root.querySelector("[data-category-chart]");
    const legend = root.querySelector("[data-category-legend]");
    const names = [
      ...categoryNames,
      ...Object.keys(categories).filter((name) => !categoryNames.includes(name)).sort(),
    ];
    chart.replaceChildren();
    legend.replaceChildren();
    if (!total) {
      chart.textContent = "No scans yet";
      chart.classList.add("category-empty");
      return;
    }
    chart.classList.remove("category-empty");
    let offset = 0;
    const segments = names.map((name, index) => {
      const count = categories[name] || 0;
      const start = offset;
      offset += count / total * 100;
      return `${colors[index % colors.length]} ${start}% ${offset}%`;
    });
    chart.style.background = `conic-gradient(${segments.join(",")})`;
    const hole = document.createElement("span");
    hole.className = "donut-hole";
    const totalLabel = document.createElement("strong");
    totalLabel.textContent = total;
    const scansLabel = document.createElement("small");
    scansLabel.textContent = "scans";
    hole.append(totalLabel, scansLabel);
    chart.append(hole);
    names.forEach((name, index) => {
      const row = document.createElement("span");
      row.className = "legend-item";
      const swatch = document.createElement("i");
      swatch.style.setProperty("--legend-color", colors[index % colors.length]);
      const title = document.createElement("span");
      title.textContent = label(name);
      const count = document.createElement("b");
      count.textContent = categories[name] || 0;
      row.append(swatch, title, count);
      legend.append(row);
    });
  }

  function drawRecent(scans) {
    const list = root.querySelector("[data-dashboard-recent]");
    list.replaceChildren();
    if (!scans.length) {
      list.innerHTML = '<div class="empty-recent"><span>✳</span><p>Your saved scans will show up here.</p><a href="/scanner">Scan your first item →</a></div>';
      return;
    }
    scans.forEach((scan) => {
      const row = document.createElement("div");
      row.className = "recent-row";
      const dot = document.createElement("span");
      dot.className = "recent-dot";
      const category = document.createElement("strong");
      category.textContent = label(scan.category);
      const date = document.createElement("span");
      date.textContent = new Date(scan.created_at).toLocaleDateString();
      const confidence = document.createElement("b");
      confidence.textContent = `${Math.round(scan.confidence * 100)}%`;
      row.append(dot, category, date, confidence);
      list.append(row);
    });
  }

  root.querySelector("[data-period]").addEventListener("change", (event) => {
    drawActivity([]);
    load(event.target.value).catch((error) => showError(error));
  });
  function showError(error) {
    root.querySelector("[data-dashboard-recent]").textContent = error.message;
  }
  load().catch(showError);
  window.addEventListener("resize", () => {
    clearTimeout(window.sortSmartResize);
    window.sortSmartResize = setTimeout(() => load().catch(showError), 150);
  });
})();
