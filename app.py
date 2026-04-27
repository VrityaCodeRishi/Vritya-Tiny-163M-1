from flask import Flask, request, jsonify, render_template_string
import torch
from classify import load_model as load_classifier, classify, cfg
from generate import GPTModel, format_prompt, generate, _ensure_weights

app = Flask(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Loading spam classifier...")
spam_model = load_classifier()
spam_model.to(device)

print("Loading instruction model...")
instruct_model = GPTModel(cfg)
instruct_model.load_state_dict(
    torch.load(_ensure_weights("instruction_finetuned.pth"),
               map_location="cpu", weights_only=True)
)
instruct_model.to(device)
instruct_model.eval()

print(f"Both models loaded on {device}")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vritya-Tiny-163M</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 16px 24px;
    background: #1e293b;
    border-bottom: 1px solid #334155;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  header h1 { font-size: 20px; font-weight: 600; color: #f8fafc; }
  .mode-toggle {
    display: flex;
    background: #0f172a;
    border-radius: 10px;
    padding: 3px;
    gap: 2px;
  }
  .mode-btn {
    padding: 8px 20px;
    border-radius: 8px;
    border: none;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    color: #94a3b8;
    background: transparent;
  }
  .mode-btn.active {
    background: #3b82f6;
    color: #fff;
  }
  .mode-btn:hover:not(.active) {
    background: #1e293b;
    color: #e2e8f0;
  }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg {
    max-width: 600px;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 15px;
    line-height: 1.6;
    animation: fadeIn 0.2s ease;
    white-space: pre-wrap;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .msg.user {
    align-self: flex-end;
    background: #3b82f6;
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.bot {
    align-self: flex-start;
    background: #1e293b;
    border: 1px solid #334155;
    border-bottom-left-radius: 4px;
  }
  .tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin-right: 6px;
  }
  .tag.ham  { background: #065f46; color: #6ee7b7; }
  .tag.spam { background: #7f1d1d; color: #fca5a5; }
  .conf { font-size: 13px; color: #94a3b8; margin-top: 4px; }
  .mode-label {
    font-size: 11px;
    color: #64748b;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  #input-area {
    padding: 16px 24px;
    background: #1e293b;
    border-top: 1px solid #334155;
  }
  #input-context {
    display: none;
    margin-bottom: 10px;
  }
  #input-context input {
    width: 100%;
    padding: 10px 14px;
    border-radius: 10px;
    border: 1px solid #334155;
    background: #0f172a;
    color: #f8fafc;
    font-size: 14px;
    outline: none;
  }
  #input-context input:focus { border-color: #3b82f6; }
  #input-context label, .params label {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 4px;
    display: block;
  }
  #gen-params {
    display: none;
    margin-bottom: 12px;
    padding: 12px 14px;
    background: #0f172a;
    border-radius: 10px;
    border: 1px solid #334155;
  }
  .params {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .param-group {
    display: flex;
    flex-direction: column;
  }
  .param-group label {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .param-group label span { color: #e2e8f0; font-weight: 500; }
  .param-group input[type=range] {
    width: 100%;
    accent-color: #3b82f6;
    margin-top: 4px;
  }
  .input-row {
    display: flex;
    gap: 10px;
  }
  .input-row input {
    flex: 1;
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid #334155;
    background: #0f172a;
    color: #f8fafc;
    font-size: 15px;
    outline: none;
    transition: border 0.15s;
  }
  .input-row input:focus { border-color: #3b82f6; }
  .input-row button {
    padding: 12px 24px;
    border-radius: 12px;
    border: none;
    background: #3b82f6;
    color: #fff;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  .input-row button:hover { background: #2563eb; }
  .input-row button:disabled { background: #475569; cursor: not-allowed; }
  .typing {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #64748b;
    animation: pulse 1s infinite;
    margin-right: 4px;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
  }
  .placeholder-text {
    color: #475569;
    font-style: italic;
    font-size: 14px;
    text-align: center;
    padding: 60px 20px;
  }
</style>
</head>
<body>
  <header>
    <h1>Vritya-Tiny-163M</h1>
    <div class="mode-toggle">
      <button class="mode-btn active" onclick="setMode('classify')" id="btn-classify">Spam Classifier</button>
      <button class="mode-btn" onclick="setMode('generate')" id="btn-generate">Instruction Following</button>
    </div>
  </header>
  <div id="chat">
    <div class="placeholder-text" id="placeholder">Type a message below to classify it as ham or spam</div>
  </div>
  <div id="input-area">
    <div id="gen-params">
      <div class="params">
        <div class="param-group">
          <label>Temperature <span id="temp-val">0.7</span></label>
          <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.7"
                 oninput="document.getElementById('temp-val').textContent=this.value">
        </div>
        <div class="param-group">
          <label>Top-k <span id="topk-val">40</span></label>
          <input type="range" id="top_k" min="1" max="100" step="1" value="40"
                 oninput="document.getElementById('topk-val').textContent=this.value">
        </div>
        <div class="param-group">
          <label>Top-p <span id="topp-val">0.9</span></label>
          <input type="range" id="top_p" min="0.1" max="1.0" step="0.05" value="0.9"
                 oninput="document.getElementById('topp-val').textContent=this.value">
        </div>
        <div class="param-group">
          <label>Repetition penalty <span id="rep-val">1.2</span></label>
          <input type="range" id="rep_penalty" min="1.0" max="2.0" step="0.1" value="1.2"
                 oninput="document.getElementById('rep-val').textContent=this.value">
        </div>
      </div>
    </div>
    <div id="input-context">
      <label>Additional context (optional)</label>
      <input id="context" type="text" placeholder="e.g. a paragraph to summarize, data to analyze...">
    </div>
    <div class="input-row">
      <input id="msg" type="text" placeholder="Type an SMS message to classify..." autocomplete="off" autofocus>
      <button id="send" onclick="send()">Send</button>
    </div>
  </div>
<script>
  const chat = document.getElementById('chat');
  const msgInput = document.getElementById('msg');
  const contextInput = document.getElementById('context');
  const contextArea = document.getElementById('input-context');
  const genParams = document.getElementById('gen-params');
  const sendBtn = document.getElementById('send');
  const placeholder = document.getElementById('placeholder');
  let mode = 'classify';

  msgInput.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

  function setMode(m) {
    mode = m;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-' + m).classList.add('active');
    if (m === 'classify') {
      msgInput.placeholder = 'Type an SMS message to classify...';
      contextArea.style.display = 'none';
      genParams.style.display = 'none';
      placeholder.textContent = 'Type a message below to classify it as ham or spam';
    } else {
      msgInput.placeholder = 'Type an instruction...';
      contextArea.style.display = 'block';
      genParams.style.display = 'block';
      placeholder.textContent = 'Type an instruction and the model will generate a response';
    }
    chat.innerHTML = '';
    chat.appendChild(placeholder);
    msgInput.focus();
  }

  function addMsg(html, cls) {
    placeholder.style.display = 'none';
    const div = document.createElement('div');
    div.className = 'msg ' + cls;
    div.innerHTML = html;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  async function send() {
    const text = msgInput.value.trim();
    if (!text) return;
    msgInput.value = '';
    addMsg(text, 'user');
    sendBtn.disabled = true;

    const loading = addMsg(
      '<span class="typing"></span><span class="typing" style="animation-delay:0.2s"></span><span class="typing" style="animation-delay:0.4s"></span>',
      'bot'
    );

    try {
      if (mode === 'classify') {
        const res = await fetch('/classify', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({text})
        });
        const data = await res.json();
        const tagCls = data.label.toLowerCase();
        loading.innerHTML =
          '<span class="tag ' + tagCls + '">' + data.label + '</span>' +
          '<div class="conf">' + (data.confidence * 100).toFixed(1) + '% confidence</div>';
      } else {
        const ctx = contextInput.value.trim();
        contextInput.value = '';
        const res = await fetch('/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            instruction: text,
            input: ctx,
            temperature: parseFloat(document.getElementById('temperature').value),
            top_k: parseInt(document.getElementById('top_k').value),
            top_p: parseFloat(document.getElementById('top_p').value),
            repetition_penalty: parseFloat(document.getElementById('rep_penalty').value),
          })
        });
        const data = await res.json();
        loading.innerHTML = data.response;
      }
    } catch {
      loading.innerHTML = 'Error processing request.';
    }
    sendBtn.disabled = false;
    msgInput.focus();
  }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/classify", methods=["POST"])
def classify_route():
    text = request.json.get("text", "")
    if not text.strip():
        return jsonify({"error": "empty message"}), 400
    label, confidence = classify(spam_model, text, device)
    return jsonify({"label": label, "confidence": confidence})


@app.route("/generate", methods=["POST"])
def generate_route():
    instruction = request.json.get("instruction", "")
    input_text = request.json.get("input", "")
    temperature = request.json.get("temperature", 0.7)
    top_k = request.json.get("top_k", 40)
    top_p = request.json.get("top_p", 0.9)
    repetition_penalty = request.json.get("repetition_penalty", 1.2)
    if not instruction.strip():
        return jsonify({"error": "empty instruction"}), 400
    prompt = format_prompt(instruction, input_text)
    response = generate(
        instruct_model, prompt, device, max_tokens=200,
        temperature=temperature, top_k=top_k,
        top_p=top_p, repetition_penalty=repetition_penalty,
    )
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=False, port=5001)
