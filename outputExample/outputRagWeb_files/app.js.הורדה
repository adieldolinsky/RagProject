(function () {
  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");
  const loadingBar = document.getElementById("loading-bar");
  const typingEl = document.getElementById("typing-indicator");
  const clearBtn = document.getElementById("clear-chat");
  const toast = document.getElementById("toast");

  let isLoading = false;

  if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 4000);
  }

  function setLoading(active) {
    isLoading = active;
    sendBtn.disabled = active;
    input.disabled = active;
    loadingBar.classList.toggle("active", active);
    typingEl.classList.toggle("active", active);
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeWelcome() {
    const welcome = messagesEl.querySelector(".welcome");
    if (welcome) welcome.remove();
  }

  function formatTime(iso) {
    if (!iso) return "";
    try {
      return new Date(iso).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "";
    }
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function renderMarkdown(text) {
    if (!text) return "";
    if (typeof marked === "undefined") {
      return escapeHtml(text);
    }
    const rawHtml = marked.parse(text);
    if (typeof DOMPurify !== "undefined") {
      return DOMPurify.sanitize(rawHtml, { USE_PROFILES: { html: true } });
    }
    return rawHtml;
  }

  function setBubbleContent(bubble, msg) {
    if (msg.role === "assistant") {
      bubble.classList.add("markdown-body");
      bubble.innerHTML = renderMarkdown(msg.content);
    } else {
      bubble.classList.remove("markdown-body");
      bubble.textContent = msg.content;
    }
  }

  function renderSources(chunks, messageId) {
    if (!chunks || !chunks.length) return "";

    const cards = chunks
      .map((chunk) => {
        const score =
          typeof chunk.score === "number"
            ? chunk.score.toFixed(3)
            : chunk.score;
        return `
          <article class="source-card">
            <header>
              <span class="badge">${escapeHtml(chunk.type || "text")}</span>
              <span>${escapeHtml(chunk.source || "document")}</span>
              <span class="score">relevance ${score}</span>
            </header>
            <div class="source-content" data-html="1"></div>
          </article>
        `;
      })
      .join("");

    return `
      <button type="button" class="sources-toggle" data-target="sources-${messageId}">
        View ${chunks.length} retrieved source${chunks.length > 1 ? "s" : ""}
      </button>
      <div class="sources-panel" id="sources-${messageId}">${cards}</div>
    `;
  }

  function renderMessage(msg) {
    const isUser = msg.role === "user";
    const id = msg.id || Date.now();
    const wrapper = document.createElement("article");
    wrapper.className = `message ${msg.role}`;
    wrapper.dataset.messageId = id;

    let sourcesHtml = "";
    if (!isUser && msg.chunks && msg.chunks.length) {
      sourcesHtml = renderSources(msg.chunks, id);
    }

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    setBubbleContent(bubble, msg);

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = formatTime(msg.created_at);

    wrapper.appendChild(bubble);
    if (sourcesHtml) {
      const temp = document.createElement("div");
      temp.innerHTML = sourcesHtml;
      while (temp.firstChild) {
        wrapper.appendChild(temp.firstChild);
      }
    }
    wrapper.appendChild(meta);

    if (msg.chunks && msg.chunks.length) {
      const panels = wrapper.querySelectorAll(".source-content");
      panels.forEach((el, i) => {
        if (msg.chunks[i]) {
          el.innerHTML = msg.chunks[i].content;
        }
      });
    }

    const toggle = wrapper.querySelector(".sources-toggle");
    if (toggle) {
      toggle.addEventListener("click", () => {
        const panel = document.getElementById(toggle.dataset.target);
        const open = panel.classList.toggle("open");
        toggle.textContent = open
          ? "Hide sources"
          : `View ${msg.chunks.length} retrieved source${msg.chunks.length > 1 ? "s" : ""}`;
      });
    }

    return wrapper;
  }

  function appendMessage(msg) {
    removeWelcome();
    messagesEl.appendChild(renderMessage(msg));
    scrollToBottom();
  }

  async function loadHistory() {
    try {
      const res = await fetch("/api/history");
      if (!res.ok) throw new Error("Failed to load history");
      const data = await res.json();
      if (data.messages && data.messages.length) {
        removeWelcome();
        data.messages.forEach((msg) =>
          messagesEl.appendChild(renderMessage(msg))
        );
        scrollToBottom();
      }
    } catch (err) {
      console.error(err);
    }
  }

  async function sendMessage(query) {
    if (!query.trim() || isLoading) return;

    appendMessage({
      role: "user",
      content: query.trim(),
      created_at: new Date().toISOString(),
    });

    input.value = "";
    input.style.height = "auto";
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim() }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Request failed");
      }

      appendMessage({
        role: "assistant",
        content: data.answer,
        chunks: data.chunks,
        created_at: data.message?.created_at || new Date().toISOString(),
        id: data.message?.id,
      });
    } catch (err) {
      showToast(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(input.value);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  });

  clearBtn.addEventListener("click", async () => {
    if (isLoading) return;
    try {
      const res = await fetch("/api/clear", { method: "POST" });
      if (!res.ok) throw new Error("Failed to clear chat");
      messagesEl.innerHTML = `
        <section class="welcome">
          <h2>Financial Document Assistant</h2>
          <p>Ask questions about the uploaded financial reports. Your conversation is saved for this session.</p>
        </section>
      `;
    } catch (err) {
      showToast(err.message);
    }
  });

  loadHistory();
})();
