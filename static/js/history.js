(() => {
  const root = document.querySelector("[data-history]");
  if (!root) return;
  const list = root.querySelector("[data-history-list]");
  const search = root.querySelector("[data-history-search]");
  const filter = root.querySelector("[data-history-filter]");
  const label = (value) => value === "uncertain" ? "Not sure yet" : value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  const date = (value) => new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  async function load() {
    list.innerHTML = '<div class="inline-loading"><span class="spinner"></span> Loading scans…</div>';
    const params = new URLSearchParams({ q: search.value, category: filter.value });
    try {
      const response = await fetch(`/api/history?${params}`);
      if (!response.ok) throw new Error("History could not be loaded.");
      const scans = await response.json();
      root.querySelector("[data-history-count]").textContent = `${scans.length} scan${scans.length === 1 ? "" : "s"}`;
      if (!scans.length) {
        list.innerHTML = '<div class="empty-history"><span>⌗</span><h2>No scans found</h2><p>Once you scan an item, its result will appear here.</p><a class="button button-green" href="/scanner">Start your first scan</a></div>';
        return;
      }
      list.replaceChildren(...scans.map((scan) => {
        const row = document.createElement("article");
        row.className = "history-item";
        const image = document.createElement("img");
        image.src = scan.image_url;
        image.alt = `Scanned ${label(scan.category).toLowerCase()}`;
        const detail = document.createElement("div");
        detail.className = "history-item-detail";
        const title = document.createElement("strong");
        title.textContent = label(scan.category);
        const time = document.createElement("span");
        time.textContent = date(scan.created_at);
        detail.append(title, time);
        const confidence = document.createElement("span");
        confidence.className = "history-confidence";
        confidence.textContent = `${Math.round(scan.confidence * 100)}% confidence`;
        const remove = document.createElement("button");
        remove.className = "delete-scan";
        remove.type = "button";
        remove.textContent = "Delete";
        remove.setAttribute("aria-label", `Delete ${label(scan.category)} scan`);
        remove.addEventListener("click", async () => {
          if (!window.confirm("Delete this scan and its saved image?")) return;
          try {
            const deletion = await fetch(`/api/history/${scan.id}`, { method: "DELETE" });
            if (!deletion.ok) throw new Error("The scan could not be deleted. Please try again.");
            await load();
          } catch (error) {
            window.alert(error.message || "The scan could not be deleted. Please try again.");
          }
        });
        row.append(image, detail, confidence, remove);
        return row;
      }));
    } catch (error) {
      list.innerHTML = "";
      const message = document.createElement("p");
      message.className = "inline-error";
      message.textContent = error.message;
      list.append(message);
    }
  }
  let timer;
  search.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(load, 180);
  });
  filter.addEventListener("change", load);
  fetch("/api/categories")
    .then((response) => {
      if (!response.ok) throw new Error("Category filters could not be loaded.");
      return response.json();
    })
    .then((data) => {
      const existing = new Set(Array.from(filter.options, (option) => option.value));
      data.categories.forEach((category) => {
        if (existing.has(category)) return;
        const option = document.createElement("option");
        option.value = category;
        option.textContent = label(category);
        filter.append(option);
      });
    })
    .catch((error) => {
      const filterError = root.querySelector("[data-filter-error]");
      filterError.textContent = error.message;
      filterError.hidden = false;
    });
  load();
})();
