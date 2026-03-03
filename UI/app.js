// ── CONFIG ──────────────────────────────────────────────────
// Change FLASK_URL if your server runs on a different port.
const FLASK_URL = "http://localhost:5000";
const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";
const CLAUDE_MODEL = "claude-sonnet-4-20250514";

// ── NAV TABS ────────────────────────────────────────────────
function showPage(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  btn.classList.add('active');
}


// ── SERVER HEALTH CHECK ─────────────────────────────────────
// Pings Flask on load so the user knows if it's running.
async function checkServer() {
  const badge = document.getElementById('server-badge');
  const dot = document.getElementById('badge-dot');
  const text = document.getElementById('badge-text');
  try {
    const res = await fetch(FLASK_URL + "/", { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      badge.className = 'badge-ok';
      dot.className = 'badge-dot dot-ok';
      text.textContent = '✅ Flask server connected — model ready';
    } else { throw new Error(); }
  } catch {
    badge.className = 'badge-error';
    dot.className = 'badge-dot dot-error';
    text.textContent = '⚠️ Flask server offline — run: python app.py';
  }
}
checkServer(); // run on page load


// ── SOIL IMAGE UPLOAD ───────────────────────────────────────
let selectedFile = null;

function handleSoilImage(event) {
  const file = event.target.files[0];
  if (!file) return;
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('soil-preview');
    preview.src = e.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);

  // Enable analyze button, hide old results
  document.getElementById('soil-btn').disabled = false;
  document.getElementById('result-section').style.display = 'none';
  document.getElementById('error-box').style.display = 'none';
}


// ── ANALYZE SOIL — calls Flask /predict ─────────────────────
async function analyzeSoil() {
  if (!selectedFile) return;

  const btn = document.getElementById('soil-btn');
  const errorBox = document.getElementById('error-box');
  const resultSec = document.getElementById('result-section');

  // Loading state
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyzing…';
  errorBox.style.display = 'none';
  resultSec.style.display = 'none';

  try {
    // Build multipart form with the image file
    const formData = new FormData();
    formData.append('image', selectedFile);

    // POST to Flask
    const response = await fetch(FLASK_URL + '/predict', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.error) throw new Error(data.error);

    // Render results
    renderResults(data);

  } catch (err) {
    errorBox.style.display = 'block';
    errorBox.innerHTML = `<strong>⚠️ Error:</strong> ${err.message}
      <br/><br/>Make sure Flask is running:  <code>python app.py</code>
      <br/>Then train the model first:  <code>python train_model.py</code>`;
  }

  btn.disabled = false;
  btn.innerHTML = '🔍 Analyze Soil';
}


// ── RENDER RESULTS ──────────────────────────────────────────
// Fills in all the result UI elements from the Flask JSON response.
function renderResults(data) {
  // Headline
  document.getElementById('res-name').textContent = data.predicted_class;
  document.getElementById('res-also').textContent = data.soil_name;
  document.getElementById('res-conf').textContent = data.confidence + '% confidence';

  // Stats
  document.getElementById('res-type').textContent = data.predicted_class;
  document.getElementById('res-ph').textContent = data.ph_range;
  document.getElementById('res-moist').textContent = data.moisture;
  document.getElementById('res-texture').textContent = data.texture;
  document.getElementById('res-organic').textContent = data.organic_matter;

  // Fertility with color
  const fertEl = document.getElementById('res-fert');
  fertEl.textContent = data.fertility;
  fertEl.className = 'stat-value fert-' + data.fertility.toLowerCase().split('/')[0].trim();

  // Crop tags
  const cropsEl = document.getElementById('res-crops');
  cropsEl.innerHTML = data.best_crops
    .map(c => `<span class="crop-tag">${c}</span>`)
    .join('');

  // Tip
  document.getElementById('res-tip').textContent = data.improvement;

  // Confidence bar chart
  const barsEl = document.getElementById('conf-bars');
  barsEl.innerHTML = Object.entries(data.all_scores)
    .sort((a, b) => b[1] - a[1])
    .map(([cls, pct]) => `
      <div class="conf-row">
        <span class="label">${cls}</span>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="pct">${pct}%</span>
      </div>
    `).join('');

  // Show result section
  document.getElementById('result-section').style.display = 'block';
  document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
}


// ── CHATBOT — calls Claude API ───────────────────────────────
let chatHistory = [];

async function sendChat() {
  const input = document.getElementById('chat-input');
  const btn = document.getElementById('chat-btn');
  const userText = input.value.trim();
  if (!userText) return;

  appendMsg('user', userText);
  input.value = '';
  btn.disabled = true;
  btn.textContent = '…';

  chatHistory.push({ role: 'user', content: userText });

  try {
    const res = await fetch(ANTHROPIC_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: CLAUDE_MODEL,
        max_tokens: 1000,
        system: `You are a friendly and expert AI agronomist specializing in 
sustainable farming, soil science, crop management, and pest control.
Give practical, simple advice that farmers can act on right away.
Use bullet points for multi-step instructions.`,
        messages: chatHistory
      })
    });

    const d = await res.json();
    if (d.error) throw new Error(d.error.message);

    const reply = d.content[0].text;
    chatHistory.push({ role: 'assistant', content: reply });
    appendMsg('bot', reply);

  } catch (err) {
    appendMsg('bot', '⚠️ ' + err.message);
  }

  btn.disabled = false;
  btn.textContent = 'Send';
}

function appendMsg(role, text) {
  const el = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}
