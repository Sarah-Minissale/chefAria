const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const messages = document.querySelector("#messages");
const submitButton = form.querySelector('button[type="submit"]');
const clearButton = document.querySelector("#clear-chat");
const promptButtons = document.querySelectorAll("[data-prompt]");

function addMessage(text, role, { pending = false } = {}) {
    const message = document.createElement("div");
    message.className = `message ${role}`;
    message.textContent = text;

    if (pending) {
        message.classList.add("is-loading");
        message.setAttribute(
            "aria-label",
            "Chef Aria di Napoli sta preparando la risposta",
        );
    }

    messages.appendChild(message);
    messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
    return message;
}

function showConversation() {
    messages.classList.add("has-conversation");
}

function setLoading(isLoading) {
    submitButton.disabled = isLoading;
    input.disabled = isLoading;
    form.setAttribute("aria-busy", String(isLoading));
}

async function sendMessage(text) {
    const message = text.trim();
    if (!message || submitButton.disabled) return;

    showConversation();
    addMessage(message, "user");
    input.value = "";
    setLoading(true);

    const placeholder = addMessage(
        "Aspetta nu poco, sto sfogliando 'o ricettario",
        "bot",
        { pending: true },
    );

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });

        let data = {};
        try {
            data = await response.json();
        } catch {
            // Il backend potrebbe restituire una risposta senza JSON.
        }

        if (!response.ok) {
            throw new Error(data.detail || "Il servizio non è disponibile.");
        }

        if (!data.answer) {
            throw new Error("La risposta dello chef è vuota.");
        }

        placeholder.classList.remove("is-loading");
        placeholder.removeAttribute("aria-label");
        placeholder.textContent = data.answer;
    } catch (error) {
        placeholder.classList.remove("is-loading");
        placeholder.classList.add("is-error");
        placeholder.removeAttribute("aria-label");
        placeholder.textContent = `Non riesco a raggiungere la cucina in questo momento. ${error.message}`;
        console.error(error);
    } finally {
        setLoading(false);
        input.focus();
        messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
    }
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(input.value);
});

promptButtons.forEach((button) => {
    button.addEventListener("click", () => {
        input.value = button.dataset.prompt;
        sendMessage(button.dataset.prompt);
    });
});

clearButton.addEventListener("click", () => {
    messages
        .querySelectorAll(".message")
        .forEach((message) => message.remove());
    messages.classList.remove("has-conversation");
    input.value = "";
    input.focus();
});
