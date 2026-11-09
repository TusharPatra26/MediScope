/* MediScope interactions — vanilla JS, no network calls (the app has no API). */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage may be unavailable */ } },
  };

  /* ---- theme ---- */
  const root = document.documentElement;
  const saved = store.get("ms-theme");
  root.dataset.theme = saved || (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const tbtn = $("#theme-toggle");
  const paint = () => { if (tbtn) { tbtn.textContent = root.dataset.theme === "dark" ? "☀" : "☾"; tbtn.setAttribute("aria-label", "Switch to " + (root.dataset.theme === "dark" ? "light" : "dark") + " theme"); } };
  paint();
  if (tbtn) tbtn.addEventListener("click", () => { root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark"; store.set("ms-theme", root.dataset.theme); paint(); document.dispatchEvent(new Event("ms:theme")); });

  /* ---- mobile sidebar ---- */
  const mt = $(".menu-toggle"), sb = $(".sidebar");
  if (mt && sb) mt.addEventListener("click", () => { const o = sb.classList.toggle("open"); mt.setAttribute("aria-expanded", o); });

  /* ---- text input: counter, example loader, dropzone ---- */
  $$("form[data-process]").forEach((form) => {
    const ta = $("textarea", form), cnt = $(".count", form);
    const upd = () => { if (ta && cnt) cnt.textContent = ta.value.length.toLocaleString() + " characters"; };
    if (ta) { ta.addEventListener("input", upd); upd(); }
    const ex = $("[data-example]", form);
    if (ex && ta) ex.addEventListener("click", () => { ta.value = ex.dataset.example; upd(); ta.focus(); });
    const dz = $(".dropzone", form), fi = dz && $("input[type=file]", dz), lab = dz && $(".dz-label", dz);
    if (dz) {
      ["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, () => dz.classList.add("drag")));
      ["dragleave", "drop"].forEach((e) => dz.addEventListener(e, () => dz.classList.remove("drag")));
      fi.addEventListener("change", () => {
        dz.classList.remove("loaded");
        if (fi.files.length) { void dz.offsetWidth; const f = fi.files[0]; lab.textContent = f.name + " · " + (f.size / 1024).toFixed(0) + " KB — will be read in memory, never stored"; dz.classList.add("loaded"); if (ta) ta.removeAttribute("required"); }
      });
    }
    /* processing overlay with a real elapsed-time counter (the server work is synchronous, so no fake percentages) */
    form.addEventListener("submit", () => {
      const ov = $("#overlay"); if (!ov) return;
      const steps = $$("input[name=steps]:checked", form).map((c) => c.dataset.label);
      const list = $("ol", ov); list.innerHTML = "";
      (steps.length ? steps : [form.dataset.process]).forEach((s) => { const li = document.createElement("li"); li.textContent = s; list.appendChild(li); });
      ov.classList.add("on"); const t0 = Date.now(); const el = $("#elapsed", ov);
      setInterval(() => { el.textContent = ((Date.now() - t0) / 1000).toFixed(0) + " s elapsed"; }, 500);
      const btn = $("button[type=submit]", form); if (btn) btn.disabled = true;
    });
  });

  /* ---- tabs ---- */
  $$("[role=tablist]").forEach((tl) => {
    const tabs = $$("[role=tab]", tl);
    tabs.forEach((t) => t.addEventListener("click", () => {
      tabs.forEach((x) => { x.setAttribute("aria-selected", x === t); const p = document.getElementById(x.getAttribute("aria-controls")); if (p) p.hidden = x !== t; });
      document.dispatchEvent(new Event("ms:redraw"));
    }));
    tl.addEventListener("keydown", (e) => { const i = tabs.indexOf(document.activeElement); if (e.key === "ArrowRight" && tabs[i + 1]) tabs[i + 1].focus(); if (e.key === "ArrowLeft" && tabs[i - 1]) tabs[i - 1].focus(); });
  });

  /* ---- entity highlighting: legend toggles + table hover ---- */
  $$("[data-toggle-ent]").forEach((b) => b.addEventListener("click", () => {
    const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", on);
    $$(".doc[data-ents]").forEach((d) => d.classList.toggle("hide-" + b.dataset.toggleEnt, !on));
  }));
  $$("tr[data-ent]").forEach((tr) => {
    const hit = (on) => $$("mark.ent").forEach((m) => { if (m.dataset.key === tr.dataset.ent) m.classList.toggle("flash", on); });
    tr.addEventListener("mouseenter", () => hit(true)); tr.addEventListener("mouseleave", () => hit(false));
    tr.addEventListener("click", () => { const m = $$("mark.ent").find((x) => x.dataset.key === tr.dataset.ent); if (m) m.scrollIntoView({ behavior: "smooth", block: "center" }); });
  });

  /* ---- classifier: sentence detail panel ---- */
  const dataEl = $("#cls-data");
  if (dataEl) {
    const sents = JSON.parse(dataEl.textContent), rows = $$(".sent[data-i]"), panel = $("#detail");
    const show = (i) => {
      const s = sents[i]; rows.forEach((r) => r.classList.toggle("sel", +r.dataset.i === i));
      $("#d-label", panel).textContent = s.label; $("#d-label", panel).className = "chip role-" + s.label;
      $("#d-score", panel).textContent = (s.score * 100).toFixed(1) + "% (uncalibrated score)";
      $("#d-text", panel).textContent = s.text;
      const alt = $("#d-alt", panel); alt.innerHTML = "";
      (s.alternatives || []).forEach((a) => { const li = document.createElement("li"); li.innerHTML = "<span>" + a.label + "</span><span>" + (a.score * 100).toFixed(1) + "%</span>"; alt.appendChild(li); });
      const ul = $("#d-terms", panel); ul.innerHTML = "";
      const terms = s.terms || [];
      if (!terms.length) ul.innerHTML = "<li class='muted'>Term attribution was computed for the first sentences only (neural model).</li>";
      const mx = Math.max(...terms.map((t) => Math.abs(t.weight)), 1e-9);
      terms.forEach((t) => { const li = document.createElement("li"); const pos = t.weight >= 0; li.innerHTML = "<span></span><span class='w'><b class='" + (pos ? "pos" : "neg") + "' style='width:" + (Math.abs(t.weight) / mx * 100).toFixed(0) + "%;" + (pos ? "left:0" : "right:0") + "'></b></span>"; li.firstChild.textContent = t.term; ul.appendChild(li); });
      if (s.distribution && window.MSCharts) MSCharts.drawProbs($("#d-chart", panel), s.distribution);
    };
    rows.forEach((r) => { r.addEventListener("click", () => show(+r.dataset.i)); r.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); show(+r.dataset.i); } }); });
    if (rows.length) show(0);
  }

  /* ---- summarization: original vs generated comparison (lexical overlap, not provenance) ---- */
  const words = (t) => new Set((t.toLowerCase().match(/[a-z0-9]{4,}/g) || []));
  const srcs = $$(".srcsent");
  $$(".sumsent").forEach((s) => s.addEventListener("click", () => {
    const sw = words(s.textContent); $$(".sumsent").forEach((x) => x.classList.toggle("on", x === s));
    const sc = srcs.map((x) => { const xw = words(x.textContent); let o = 0; xw.forEach((w) => { if (sw.has(w)) o++; }); return o / Math.max(1, Math.min(sw.size, xw.size)); });
    const top = [...sc].sort((a, b) => b - a)[2] || 0.2; srcs.forEach((x, i) => x.classList.toggle("hit", sc[i] >= Math.max(top, 0.25)));
    const first = srcs.find((x) => x.classList.contains("hit")); if (first) first.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }));

  /* ---- overview specimen: filter sentences by role ---- */
  $$(".rolebtn").forEach((b) => b.addEventListener("click", () => {
    b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") !== "true");
    const on = new Set($$(".rolebtn[aria-pressed=true]").map((x) => x.dataset.role));
    $$(".sheet-body li").forEach((li) => li.classList.toggle("dim", !on.has(li.dataset.role)));
  }));

  /* ---- confirm destructive actions ---- */
  $$("form[data-confirm]").forEach((f) => f.addEventListener("submit", (e) => { if (!confirm(f.dataset.confirm)) e.preventDefault(); }));
})();
