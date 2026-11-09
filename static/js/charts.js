/* Chart.js helpers (vendored locally). Charts read colours from CSS variables so they follow the theme. */
(function () {
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const palette = () => [css("--forest"), css("--terracotta"), css("--ochre"), css("--sage"), css("--stone"), css("--charcoal")];
  const roleColor = (l) => css("--r-" + l.toLowerCase());
  const registry = [];

  function base() {
    if (!window.Chart) return null;
    Chart.defaults.font.family = '"IBM Plex Sans", sans-serif';
    Chart.defaults.font.size = 11;
    Chart.defaults.color = css("--muted");
    Chart.defaults.borderColor = css("--line");
    return true;
  }

  function drawProbs(canvas, dist) {
    if (!base()) return;
    if (canvas._chart) canvas._chart.destroy();
    const labels = Object.keys(dist);
    canvas._chart = new Chart(canvas, {
      type: "bar",
      data: { labels, datasets: [{ data: labels.map((l) => dist[l]), backgroundColor: labels.map(roleColor), borderWidth: 0 }] },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
        scales: { x: { min: 0, max: 1, grid: { color: css("--line") }, ticks: { callback: (v) => Math.round(v * 100) + "%" } }, y: { grid: { display: false } } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => (c.parsed.x * 100).toFixed(1) + "% (uncalibrated score)" } } } },
    });
    return canvas._chart;
  }

  function build(canvas) {
    if (!base()) return;
    const cfg = JSON.parse(canvas.dataset.config);
    const p = palette();
    (cfg.data.datasets || []).forEach((d, i) => {
      if (cfg.roleColors) d.backgroundColor = cfg.data.labels.map(roleColor);
      else if (!d.backgroundColor) d.backgroundColor = cfg.type === "scatter" ? p[i % p.length] : p[i % p.length];
      if (cfg.type === "scatter" && !d.pointRadius) d.pointRadius = 6;
    });
    cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: { duration: 350 } }, cfg.options || {});
    if (canvas._chart) canvas._chart.destroy();
    canvas._chart = new Chart(canvas, cfg);
    registry.push(canvas);
  }

  function initAll() { document.querySelectorAll("canvas[data-config]").forEach(build); }
  window.MSCharts = { drawProbs, initAll, roleColor };
  document.addEventListener("DOMContentLoaded", initAll);
  document.addEventListener("ms:theme", () => { document.querySelectorAll("canvas[data-config]").forEach(build); document.dispatchEvent(new Event("ms:redraw")); });
})();
