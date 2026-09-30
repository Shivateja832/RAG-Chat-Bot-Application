const form = document.querySelector('#chat-form');
const input = document.querySelector('#query-input');
const messages = document.querySelector('#messages');
const welcome = document.querySelector('#welcome');
const sendButton = document.querySelector('#send-button');
const ingestButton = document.querySelector('#ingest-button');
const ingestLabel = document.querySelector('#ingest-label');
const toast = document.querySelector('#toast');
const history = [];
let busy = false;
let indexReady = false;
let toastTimer;
let requestTimer;
let requestStartedAt = 0;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('visible');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove('visible'), 4300);
}

function setBusy(value) {
  busy = value;
  updateChatControls();
  window.clearInterval(requestTimer);
  if (value) {
    requestStartedAt = Date.now();
    updateRequestLabel();
    requestTimer = window.setInterval(updateRequestLabel, 1000);
  } else {
    sendButton.innerHTML = 'Ask the book <span aria-hidden="true">↗</span>';
  }
}

function updateRequestLabel() {
  const elapsedSeconds = Math.floor((Date.now() - requestStartedAt) / 1000);
  sendButton.textContent = `Thinking · ${elapsedSeconds}s`;
}

function updateChatControls() {
  sendButton.disabled = busy || !indexReady;
  input.disabled = busy || !indexReady;
  document.querySelectorAll('.prompt-link').forEach((button) => {
    button.disabled = busy || !indexReady;
  });
}

function addQuestion(text) {
  const message = document.createElement('div');
  message.className = 'message message-question';
  message.textContent = text;
  messages.append(message);
  welcome.hidden = true;
  return message;
}

function addTyping() {
  const message = document.createElement('div');
  message.className = 'message answer-message';
  message.innerHTML = '<div class="answer-header"><span class="answer-mark">f</span> SEARCHING THE SOURCE</div><div class="typing" aria-label="Searching"><span></span><span></span><span></span></div>';
  messages.append(message);
  return message;
}

function addAnswer(payload) {
  const article = document.createElement('article');
  article.className = 'message answer-message';
  const header = document.createElement('div');
  header.className = 'answer-header';
  header.innerHTML = '<span class="answer-mark">f</span> FROM THE EBOOK';
  const body = document.createElement('div');
  body.className = 'answer-body';
  body.textContent = payload.final_answer;
  if (payload.confidence_score === 0) body.classList.add('is-refusal');
  const tools = document.createElement('div');
  tools.className = 'answer-tools';
  const label = document.createElement('span');
  label.textContent = `RETRIEVAL CONFIDENCE ${Math.round(payload.confidence_score * 100)}%`;
  const track = document.createElement('span');
  track.className = 'confidence-track';
  const fill = document.createElement('span');
  fill.style.width = `${Math.round(payload.confidence_score * 100)}%`;
  track.append(fill);
  tools.append(label, track);
  article.append(header, body, tools);

  if (payload.retrieved_context_chunks.length) {
    const details = document.createElement('details');
    details.className = 'context-details';
    const summary = document.createElement('summary');
    summary.textContent = `Evidence · ${payload.retrieved_context_chunks.length} source ${payload.retrieved_context_chunks.length === 1 ? 'passage' : 'passages'}`;
    details.append(summary);
    for (const chunk of payload.retrieved_context_chunks) {
      const evidence = document.createElement('div');
      evidence.className = 'context-chunk';
      const page = chunk.match(/^\[Page (\d+)\]\s*/);
      if (page) {
        const pageLabel = document.createElement('span');
        pageLabel.className = 'context-page';
        pageLabel.textContent = `PAGE ${page[1]}`;
        evidence.append(pageLabel);
        evidence.append(document.createTextNode(chunk.slice(page[0].length)));
      } else {
        evidence.textContent = chunk;
      }
      details.append(evidence);
    }
    article.append(details);
  }
  messages.append(article);
  return article;
}

async function ask(query) {
  const cleanQuery = query.trim();
  if (!cleanQuery || busy || !indexReady) return;
  addQuestion(cleanQuery);
  input.value = '';
  resizeInput();
  setBusy(true);
  const pending = addTyping();
  messages.scrollIntoView({ behavior: 'smooth', block: 'end' });
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: cleanQuery, history: history.slice(-8) }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : payload.detail?.message;
      throw new Error(detail || 'The request could not be completed.');
    }
    pending.remove();
    addAnswer(payload);
    history.push({ role: 'user', content: cleanQuery }, { role: 'assistant', content: payload.final_answer });
  } catch (error) {
    pending.remove();
    addQuestionError(error.message);
  } finally {
    setBusy(false);
    input.focus();
    messages.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }
}

function addQuestionError(message) {
  const article = document.createElement('article');
  article.className = 'message answer-message';
  const body = document.createElement('div');
  body.className = 'answer-body is-refusal';
  body.textContent = message;
  article.append(body);
  messages.append(article);
}

async function refreshStatus() {
  const dot = document.querySelector('#source-dot');
  const stateText = document.querySelector('#source-state');
  const countText = document.querySelector('#index-count');
  const meter = document.querySelector('#index-meter-fill');
  const indexLabel = document.querySelector('#index-label');
  try {
    const response = await fetch('/api/status');
    const data = await response.json();
    const labels = {
      ready: 'Index ready',
      needs_ingestion: 'Needs indexing',
      missing_models: 'Models needed',
      ollama_offline: 'Start Ollama',
      missing_pinecone_key: 'Add Pinecone key',
      pinecone_auth_error: 'Check Pinecone key',
      connection_error: 'Connection issue',
      checking: 'Checking index',
    };
    indexReady = data.state === 'ready';
    updateChatControls();
    stateText.textContent = labels[data.state] || 'Connection issue';
    indexLabel.textContent = data.state === 'ready' ? 'Indexed' : data.state === 'needs_ingestion' ? 'Not indexed' : ['missing_models', 'missing_pinecone_key'].includes(data.state) ? 'Setup needed' : data.state === 'ollama_offline' ? 'Offline' : 'Unavailable';
    countText.textContent = Number(data.vector_count || 0).toLocaleString();
    meter.style.width = data.state === 'ready' ? '100%' : '0%';
    dot.classList.toggle('ready', data.state === 'ready');
    dot.classList.toggle('error', ['connection_error', 'ollama_offline', 'pinecone_auth_error'].includes(data.state));
    document.querySelector('#index-name').textContent = data.index_name || 'agentic-ai-ebook-nomic';
    if (data.state === 'missing_models') {
      const commands = (data.missing_models || []).map((model) => `ollama pull ${model}`).join(' · ');
      document.querySelector('#composer-hint').textContent = `Download the local models: ${commands}`;
    } else if (data.state === 'ollama_offline') {
      document.querySelector('#composer-hint').textContent = 'Open the Ollama app, then refresh this page.';
    } else if (data.state === 'missing_pinecone_key') {
      document.querySelector('#composer-hint').textContent = 'Add PINECONE_API_KEY to .env, then restart the app.';
    } else if (data.state === 'pinecone_auth_error') {
      document.querySelector('#composer-hint').textContent = 'Pinecone rejected its key. Update PINECONE_API_KEY in .env.';
    } else if (data.state === 'needs_ingestion') {
      document.querySelector('#composer-hint').textContent = 'Build the ebook index first; chat unlocks when passages are ready.';
    } else if (data.state === 'connection_error') {
      document.querySelector('#composer-hint').textContent = 'Cannot reach Pinecone. Check your network and Pinecone configuration.';
    } else {
      document.querySelector('#composer-hint').textContent = 'Answers cite the pages they come from.';
    }
  } catch {
    indexReady = false;
    updateChatControls();
    stateText.textContent = 'API unavailable';
    dot.classList.add('error');
    indexLabel.textContent = 'Offline';
    document.querySelector('#composer-hint').textContent = 'Start the local app, then refresh this page.';
  }
}

function resizeInput() {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  ask(input.value);
});
input.addEventListener('input', resizeInput);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
document.querySelectorAll('.prompt-link').forEach((button) => {
  button.addEventListener('click', () => ask(button.dataset.query || ''));
});
document.querySelector('#clear-button').addEventListener('click', () => {
  messages.replaceChildren();
  history.length = 0;
  welcome.hidden = false;
  input.focus();
});
ingestButton.addEventListener('click', async () => {
  if (busy) return;
  ingestButton.disabled = true;
  ingestButton.classList.add('is-loading');
  ingestLabel.textContent = 'Downloading & indexing…';
  try {
    const response = await fetch('/api/ingest', { method: 'POST' });
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : data.detail?.message;
      throw new Error(detail || 'Indexing could not be completed.');
    }
    showToast(`Indexed ${data.chunks_indexed} passages across ${data.pages_read} pages.`);
    await refreshStatus();
  } catch (error) {
    showToast(error.message);
  } finally {
    ingestButton.disabled = false;
    ingestButton.classList.remove('is-loading');
    ingestLabel.textContent = 'Build / refresh index';
  }
});

refreshStatus();
