/* Enquiry page. Sends the question to POST /api/enquire and renders the reply.
   The backend decides every answer and fallback; this file only displays them.
   All server and user text is inserted with textContent, never innerHTML. */
(function () {
  'use strict';

  var app = document.getElementById('chat-app');
  if (!app) { return; }

  var apiUrl = app.getAttribute('data-api-url');
  var log = document.getElementById('chat-log');
  var form = document.getElementById('chat-form');
  var input = document.getElementById('question-input');
  var sendButton = document.getElementById('send-button');
  var resetButton = document.getElementById('reset-button');
  var welcomeTemplate = document.getElementById('welcome-template');
  var chips = Array.prototype.slice.call(document.querySelectorAll('.chip'));

  var maxLength = parseInt(input.getAttribute('maxlength'), 10) || 500;
  var REQUEST_TIMEOUT_MS = 15000;
  var busy = false;
  var typingRow = null;
  var controller = null;
  var requestId = 0;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  function scrollToEnd() { log.scrollTop = log.scrollHeight; }

  function addRow(kind, bubble) {
    var row = el('div', 'msg msg-' + kind);
    row.appendChild(bubble);
    log.appendChild(row);
    scrollToEnd();
    return row;
  }

  function addUserMessage(text) {
    var bubble = el('div', 'bubble bubble-user');
    bubble.appendChild(el('p', null, text));
    addRow('user', bubble);
  }

  function addErrorMessage(text) {
    var bubble = el('div', 'bubble bubble-bot bubble-error');
    bubble.setAttribute('role', 'alert');
    bubble.appendChild(el('p', null, text));
    addRow('bot', bubble);
  }

  function showTyping() {
    var bubble = el('div', 'bubble bubble-bot');
    var dots = el('span', 'typing');
    dots.setAttribute('role', 'status');
    dots.setAttribute('aria-label', 'The assistant is searching the FAQ');
    dots.appendChild(el('span'));
    dots.appendChild(el('span'));
    dots.appendChild(el('span'));
    bubble.appendChild(dots);
    typingRow = addRow('bot', bubble);
  }

  function hideTyping() {
    if (typingRow && typingRow.parentNode) { typingRow.parentNode.removeChild(typingRow); }
    typingRow = null;
  }

  function renderMatch(data) {
    var bubble = el('div', 'bubble bubble-bot');
    bubble.appendChild(el('p', 'bubble-title', data.message || 'Here is the information available in our college FAQ.'));

    var head = el('div', 'answer-head');
    head.appendChild(el('span', 'badge', 'FAQ ID: ' + data.faq_id));
    if (data.category) { head.appendChild(el('span', 'tag', data.category)); }
    bubble.appendChild(head);

    if (data.question) {
      var matched = el('p', null);
      matched.appendChild(el('strong', null, 'Matched question: '));
      matched.appendChild(document.createTextNode(data.question));
      bubble.appendChild(matched);
    }
    bubble.appendChild(el('p', 'answer-body', data.answer));

    var percent = Math.max(0, Math.min(100, Math.round((Number(data.similarity) || 0) * 100)));
    var confidence = el('div', 'confidence');
    confidence.setAttribute('title', 'Text similarity between your question and this FAQ. It is not a probability of being correct.');
    confidence.appendChild(el('span', null, 'Match confidence'));
    var meter = el('span', 'meter');
    meter.setAttribute('role', 'progressbar');
    meter.setAttribute('aria-label', 'Match confidence');
    meter.setAttribute('aria-valuemin', '0');
    meter.setAttribute('aria-valuemax', '100');
    meter.setAttribute('aria-valuenow', String(percent));
    var fill = el('span', 'meter-fill');
    meter.appendChild(fill);
    confidence.appendChild(meter);
    confidence.appendChild(el('span', null, percent + '%'));
    bubble.appendChild(confidence);

    addRow('bot', bubble);
    window.requestAnimationFrame(function () { fill.style.width = percent + '%'; });
  }

  function renderFallback(data) {
    var bubble = el('div', 'bubble bubble-bot');
    bubble.appendChild(el('p', null, data.answer || data.message));
    addRow('bot', bubble);
  }

  function setBusy(state) {
    busy = state;
    input.disabled = state;
    sendButton.disabled = state;
    chips.forEach(function (chip) { chip.disabled = state; });
    if (state) { sendButton.textContent = 'Sending'; } else { sendButton.textContent = 'Send'; input.focus(); }
  }

  function parseJson(response) {
    return response.json().catch(function () { return null; });
  }

  function ask(rawQuestion) {
    if (busy) { return; }
    var question = (rawQuestion || '').trim();
    if (!question) {
      addErrorMessage('Please enter a question.');
      input.focus();
      return;
    }
    if (question.length > maxLength) {
      addErrorMessage('Please keep your question under ' + maxLength + ' characters.');
      return;
    }

    addUserMessage(question);
    input.value = '';
    setBusy(true);
    showTyping();

    var myRequest = ++requestId;
    controller = new AbortController();
    var timer = window.setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS);

    fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ question: question }),
      signal: controller.signal
    })
      .then(function (response) {
        return parseJson(response).then(function (data) { return { ok: response.ok, data: data }; });
      })
      .then(function (result) {
        if (myRequest !== requestId) { return; }
        hideTyping();
        var data = result.data;
        if (!data) {
          addErrorMessage('The server sent an unexpected reply. Please try again.');
        } else if (!result.ok || !data.success) {
          addErrorMessage(data.error || data.message || 'Something went wrong. Please try again.');
        } else if (data.matched) {
          renderMatch(data);
        } else {
          renderFallback(data);
        }
      })
      .catch(function (error) {
        if (myRequest !== requestId) { return; }
        hideTyping();
        if (error && error.name === 'AbortError') {
          addErrorMessage('The request took too long. Please try again.');
        } else {
          addErrorMessage('I could not reach the server. Please check your connection and try again.');
        }
      })
      .then(function () {
        window.clearTimeout(timer);
        if (myRequest === requestId) { setBusy(false); }
      });
  }

  function resetConversation() {
    requestId += 1;
    if (controller) { controller.abort(); }
    hideTyping();
    log.textContent = '';
    if (welcomeTemplate && welcomeTemplate.content) {
      log.appendChild(welcomeTemplate.content.cloneNode(true));
    }
    input.value = '';
    setBusy(false);
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    ask(input.value);
  });

  chips.forEach(function (chip) {
    chip.addEventListener('click', function () { ask(chip.getAttribute('data-question')); });
  });

  resetButton.addEventListener('click', resetConversation);

  resetConversation();
})();
