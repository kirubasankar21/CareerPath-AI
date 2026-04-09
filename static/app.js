/**
 * Minimal client logic: honor prefers-reduced-motion by showing content immediately.
 */
(function () {
  var mq =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)");
  if (mq && mq.matches) {
    document.querySelectorAll(".animate-in").forEach(function (el) {
      el.classList.add("reveal");
    });
  }
})();
