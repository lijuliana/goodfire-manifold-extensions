
function renderFig(el, fig){
  Plotly.newPlot(el, fig.data, fig.layout, {responsive:true, displaylogo:false,
    modeBarButtonsToRemove:['toImage','resetCameraLastSave3d','hoverClosest3d']});
}
function figFrom(id){ return JSON.parse(document.getElementById(id).textContent); }
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.eager-fig').forEach(el => renderFig(el, figFrom(el.dataset.fig)));
  const lazy = document.querySelectorAll('.lazy-fig');
  if (!('IntersectionObserver' in window)) { lazy.forEach(el => renderFig(el, figFrom(el.dataset.fig))); return; }
  const io = new IntersectionObserver(entries => entries.forEach(e => {
    if (e.isIntersecting) { renderFig(e.target, figFrom(e.target.dataset.fig)); io.unobserve(e.target); }
  }), {rootMargin: '250px'});
  lazy.forEach(el => io.observe(el));
});
