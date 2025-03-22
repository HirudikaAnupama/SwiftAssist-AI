document.addEventListener("DOMContentLoaded", function () {
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const messagesDiv = document.getElementById("messages");

  // Function to append a message to the chat window
  function appendMessage(className, htmlContent) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", className);

    // Create text bubble
    const bubble = document.createElement("div");
    bubble.classList.add("text");
    // Use innerHTML to render HTML tags (bold, <br>, etc.)
    bubble.innerHTML = htmlContent;

    messageDiv.appendChild(bubble);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Auto-scroll
  }

  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // Append user message (plain text)
    appendMessage("user-message", query);
    userInput.value = "";

    // Fetch chatbot response from the server
    fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query })
    })
      .then(response => response.json())
      .then(data => {
        // Show each piece in bold with a line break
        const botResponse = `
<strong>Intent:</strong> ${data["Predicted Intent"]}<br>
<strong>Type:</strong> ${data["Intent Type"]}<br>
<strong>Sentiment:</strong> ${data["Sentiment"]}<br>
<strong>Response:</strong> ${data["Final Response"]}
        `;
        appendMessage("bot-message", botResponse);
      })
      .catch(error => {
        console.error("Error:", error);
        appendMessage("bot-message", "Sorry, something went wrong.");
      });
  });
});
