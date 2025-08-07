// JavaScript to connect to the WebSocket and display subtitles
document.getElementById('start-btn').addEventListener('click', function() {
  const urlInput = document.getElementById('stream-url');
  const streamUrl = urlInput.value.trim();
  if (!streamUrl) {
    alert('Please enter a livestream URL.');
    return;
  }
  const subtitlesDiv = document.getElementById('subtitles');
  subtitlesDiv.innerHTML = '';
  // Open WebSocket connection
  const ws = new WebSocket(`ws://${window.location.host}/ws`);
  ws.onopen = function() {
    // Send the stream URL as JSON
    ws.send(JSON.stringify({ url: streamUrl }));
  };
  ws.onerror = function(err) {
    console.error('WebSocket error:', err);
  };
  ws.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);
      if (data.error) {
        subtitlesDiv.innerText = 'Error: ' + data.error;
        return;
      }
      // Append new subtitle line
      const line = document.createElement('div');
      line.textContent = `${data.index}: ${data.english}`;
      subtitlesDiv.appendChild(line);
      subtitlesDiv.scrollTop = subtitlesDiv.scrollHeight;
    } catch (e) {
      console.error('Invalid JSON:', event.data);
    }
  };
});