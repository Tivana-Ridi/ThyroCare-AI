// static/js/main.js
// Global JS for ThyroCare-AI (you can add more later)

// Example: smooth scroll for anchor links (optional)
document.addEventListener("click", function (e) {
  if (e.target.matches("a[href^='#']")) {
    e.preventDefault();
    const targetId = e.target.getAttribute("href").substring(1);
    const target = document.getElementById(targetId);
    if (target) {
      target.scrollIntoView({ behavior: "smooth" });
    }
  }
});
