const fileInput = document.querySelector('#file-input');
const browseButton = document.querySelector('#browse-button');
const dropZone = document.querySelector('#drop-zone');
const fileList = document.querySelector('#file-list');
const askForm = document.querySelector('#ask-form');
const questionInput = document.querySelector('#question');
const messages = document.querySelector('#messages');
const emptyState = document.querySelector('#empty-state');
const sendButton = document.querySelector('.send-button');

browseButton.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (event) => {
  if (!event.target.closest('button')) fileInput.click();
});
['dragenter', 'dragover'].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', event => uploadFile(event.dataTransfer.files[0]));
fileInput.addEventListener('change', event => uploadFile(event.target.files[0]));

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.txt')) {
    addFile(file.name, 'Only UTF-8 .txt files are supported', true);
    return;
  }
  addFile(file.name, 'Uploading…');
  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await fetch('/documents/upload', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Upload failed');
    updateLastFile(`${data.chunks_added} chunk${data.chunks_added === 1 ? '' : 's'} ready`);
    document.querySelector('#connection-label').textContent = 'Document ready';
  } catch (error) {
    updateLastFile(error.message, true);
  } finally { fileInput.value = ''; }
}

function addFile(name, status, isError = false) {
  const item = document.createElement('div'); item.className = 'file-item';
  item.innerHTML = `<span class="file-badge">TXT</span><div class="file-meta"><div class="file-name"></div><div class="file-status${isError ? ' error' : ''}"></div></div>`;
  item.querySelector('.file-name').textContent = name;
  item.querySelector('.file-status').textContent = status;
  fileList.appendChild(item);
}
function updateLastFile(status, isError = false) {
  const statusNode = fileList.lastElementChild?.querySelector('.file-status');
  if (statusNode) { statusNode.textContent = status; statusNode.classList.toggle('error', isError); }
}

askForm.addEventListener('submit', async event => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || sendButton.disabled) return;
  emptyState?.remove();
  addMessage('You', question, true);
  questionInput.value = ''; resizeInput();
  const typing = addMessage('RAGbot', 'Thinking…', false, true);
  sendButton.disabled = true;
  try {
    const response = await fetch('/ask', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ question }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The question could not be answered');
    typing.remove(); addMessage('RAGbot', data.answer, false, false, data.sources || []);
  } catch (error) { typing.remove(); addMessage('RAGbot', error.message, false, false, [], true); }
  finally { sendButton.disabled = false; questionInput.focus(); }
});

function addMessage(label, text, isUser, typing = false, sources = [], isError = false) {
  const message = document.createElement('article'); message.className = `message ${isUser ? 'user' : 'bot'}`;
  const bodyClass = `message-body${isError ? ' error' : ''}${typing ? ' typing' : ''}`;
  message.innerHTML = `<div class="message-label">${label}</div><div class="${bodyClass}"></div>`;
  message.querySelector('.message-body').textContent = text;
  if (sources.length) {
    const sourceWrap = document.createElement('div'); sourceWrap.className = 'sources';
    sources.forEach(source => { const tag = document.createElement('span'); tag.className = 'source'; tag.textContent = `${source.filename} · chunk ${source.chunk_index}`; sourceWrap.appendChild(tag); });
    message.appendChild(sourceWrap);
  }
  messages.appendChild(message); messages.scrollTop = messages.scrollHeight; return message;
}

questionInput.addEventListener('input', resizeInput);
questionInput.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); askForm.requestSubmit(); } });
function resizeInput() { questionInput.style.height = 'auto'; questionInput.style.height = `${Math.min(questionInput.scrollHeight, 130)}px`; }
