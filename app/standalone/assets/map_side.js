/* The map's side (style.css reads --map-side): a square of MAP_SHARE of the window's area, never below MAP_MIN
   pixels. CSS cannot take the square root of the window's area, so Dash serves this with the page's assets. */
const MAP_SHARE = 0.125;
const MAP_MIN = 240;

function setMapSide() {
  const side = Math.max(MAP_MIN, Math.sqrt(MAP_SHARE * window.innerWidth * window.innerHeight));
  document.documentElement.style.setProperty('--map-side', `${Math.round(side)}px`);
}

window.addEventListener('resize', setMapSide);
setMapSide();
