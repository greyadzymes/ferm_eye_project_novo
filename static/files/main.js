const plotImg = document.getElementById('plot-img');

const evtSource = new EventSource('/plot');
evtSource.onmessage = (event) => {
  plotImg.src = URL.createObjectURL(new Blob([event.data], {type: 'image/png'}));
};